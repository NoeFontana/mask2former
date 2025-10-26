"""Unit tests for Mask2Former decoder components."""

import pytest
import torch

from mask2former.modeling.attn import CrossAttention, SelfAttention
from mask2former.modeling.decoder import DecoderLayer


@pytest.fixture(scope="class")
def decoder_layer() -> DecoderLayer:
    """Standard decoder layer for testing."""
    return DecoderLayer(embedding_dim=256, num_head=8, hidden_dim=1024)


@pytest.fixture(scope="class")
def test_generator() -> torch.Generator:
    """Seeded generator for reproducible test data."""
    generator = torch.Generator()
    generator.manual_seed(42)
    return generator


@pytest.fixture(scope="class")
def sample_inputs(test_generator: torch.Generator) -> dict[str, torch.Tensor]:
    """Sample inputs with seeded generator for reproducibility."""
    batch_size, num_queries, embedding_dim = 2, 100, 256
    height, width = 32, 32

    return {
        "query_features": torch.randn(
            batch_size, num_queries, embedding_dim, generator=test_generator
        ),
        "image_features": torch.randn(
            batch_size, embedding_dim, height, width, generator=test_generator
        ),
        "masks": torch.randint(
            0, 2, (batch_size, num_queries, height, width), generator=test_generator
        ).bool(),
        "pos_query_embeddings": torch.randn(
            batch_size, num_queries, embedding_dim, generator=test_generator
        ),
        "pos_image_embeddings": torch.randn(
            batch_size, height * width, embedding_dim, generator=test_generator
        ),
    }


class TestDecoderLayer:
    """Tests for DecoderLayer."""

    def test_initialization(self, decoder_layer: DecoderLayer) -> None:
        """Test proper initialization of DecoderLayer."""
        assert isinstance(decoder_layer.cross_attention, CrossAttention)
        assert isinstance(decoder_layer.self_attention, SelfAttention)
        assert decoder_layer.cross_attention.num_head == 8
        assert decoder_layer.self_attention.num_head == 8

    def test_forward_shape(
        self, decoder_layer: DecoderLayer, sample_inputs: dict[str, torch.Tensor]
    ) -> None:
        """Test output shape is correct."""
        output = decoder_layer(**sample_inputs)
        expected_shape = sample_inputs["query_features"].shape
        assert output.shape == expected_shape
        assert output.dtype == torch.float32

    def test_gradient_flow(
        self, decoder_layer: DecoderLayer, sample_inputs: dict[str, torch.Tensor]
    ) -> None:
        """Test gradients flow through the layer."""
        # Create fresh copies to avoid modifying the fixture for other tests
        inputs_copy = {key: tensor.clone() for key, tensor in sample_inputs.items()}

        # Only set requires_grad for floating point tensors
        for tensor in inputs_copy.values():
            if isinstance(tensor, torch.Tensor) and tensor.dtype.is_floating_point:
                tensor.requires_grad_(True)

        output = decoder_layer(**inputs_copy)
        loss = output.sum()
        loss.backward()

        # Check gradients exist for input tensors (only floating point)
        assert inputs_copy["query_features"].grad is not None
        assert inputs_copy["image_features"].grad is not None
        assert inputs_copy["pos_query_embeddings"].grad is not None
        assert inputs_copy["pos_image_embeddings"].grad is not None

    def test_residual_connections(self, sample_inputs: dict[str, torch.Tensor]) -> None:
        """Test residual connections work correctly."""
        layer = DecoderLayer(embedding_dim=256, num_head=8, hidden_dim=1024)

        # Zero out attention and FFN to test pure residual
        with torch.no_grad():
            for param in layer.cross_attention.parameters():
                param.zero_()
            for param in layer.self_attention.parameters():
                param.zero_()
            for param in layer.ffn.parameters():
                param.zero_()

        output = layer(**sample_inputs)

        # With zero weights, output should be close to layer-normalized input
        expected = layer.norm_3(
            layer.norm_2(layer.norm_1(sample_inputs["query_features"]))
        )
        torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-6)

    @pytest.mark.parametrize("embedding_dim", [64, 128, 256, 512])
    @pytest.mark.parametrize("num_head", [1, 4, 8])
    def test_different_dimensions(self, embedding_dim: int, num_head: int) -> None:
        """Test with different embedding dimensions and head counts."""
        if embedding_dim % num_head != 0:
            pytest.skip("embedding_dim must be divisible by num_head")

        layer = DecoderLayer(
            embedding_dim=embedding_dim, num_head=num_head, hidden_dim=embedding_dim * 4
        )

        batch_size, num_queries = 1, 50
        inputs = {
            "query_features": torch.randn(batch_size, num_queries, embedding_dim),
            "image_features": torch.randn(batch_size, embedding_dim, 16, 16),
            "masks": torch.ones(batch_size, num_queries, 16, 16).bool(),
            "pos_query_embeddings": torch.randn(batch_size, num_queries, embedding_dim),
            "pos_image_embeddings": torch.randn(batch_size, 256, embedding_dim),
        }

        output = layer(**inputs)
        assert output.shape == (batch_size, num_queries, embedding_dim)

    def test_torch_compile_compatibility(
        self, decoder_layer: DecoderLayer, sample_inputs: dict[str, torch.Tensor]
    ) -> None:
        """Test that DecoderLayer is compatible with torch.compile."""
        compiled_layer = torch.compile(decoder_layer, mode="reduce-overhead")

        # Run compiled version
        compiled_output = compiled_layer(**sample_inputs)

        # Run original version
        original_output = decoder_layer(**sample_inputs)

        # Outputs should be very close
        torch.testing.assert_close(
            compiled_output, original_output, atol=1e-5, rtol=1e-5
        )
