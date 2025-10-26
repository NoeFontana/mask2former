"""Unit tests for Mask2Former decoder components."""

import logging
import time

import pytest
import torch

from mask2former.modeling.attn import CrossAttention, SelfAttention
from mask2former.modeling.decoder import DecoderLayer, TransformerDecoder

logger = logging.getLogger(__name__)


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


def _benchmark_execution(
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    num_runs: int,
) -> list[float]:
    """Helper method to benchmark model execution with proper GPU synchronization."""
    # Detect if we're using GPU
    device = next(model.parameters()).device
    use_cuda = device.type == "cuda"

    times = []
    for _ in range(num_runs):
        if use_cuda:
            # Use CUDA events for accurate GPU timing
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            torch.cuda.synchronize()
            start_event.record()
            _ = model(**inputs)
            end_event.record()
            torch.cuda.synchronize()

            elapsed = (
                start_event.elapsed_time(end_event) / 1000.0
            )  # Convert ms to seconds
        else:
            # Use CPU timing
            start = time.perf_counter()
            _ = model(**inputs)
            elapsed = time.perf_counter() - start

        times.append(elapsed)
    return times


def _calculate_stats(times: list[float]) -> dict[str, float]:
    """Helper method to calculate timing statistics."""
    return {
        "avg": sum(times) / len(times),
        "min": min(times),
        "max": max(times),
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


@pytest.fixture(scope="class")
def transformer_decoder() -> TransformerDecoder:
    """Standard transformer decoder for testing."""
    return TransformerDecoder(
        num_layers=2,
        embedding_dim=256,
        num_query=100,
        num_head=8,
        hidden_dim=1024,
        num_feature_levels=3,
        input_size=(64, 64),
        output_divisors=(8, 4, 2),
    )


@pytest.fixture(scope="class")
def transformer_inputs(
    test_generator: torch.Generator,
) -> dict[str, torch.Tensor | list[torch.Tensor]]:
    """Sample inputs for TransformerDecoder with seeded generator."""
    batch_size = 2
    embedding_dim = 256

    # Multi-scale feature maps at different resolutions
    image_features_list = [
        torch.randn(batch_size, embedding_dim, 8, 8, generator=test_generator),
        torch.randn(batch_size, embedding_dim, 16, 16, generator=test_generator),
        torch.randn(batch_size, embedding_dim, 32, 32, generator=test_generator),
    ]

    mask_features = torch.randn(
        batch_size, embedding_dim, 32, 32, generator=test_generator
    )

    return {
        "image_features_list": image_features_list,
        "mask_features": mask_features,
    }


class TestTransformerDecoder:
    """Tests for TransformerDecoder."""

    def test_initialization(self, transformer_decoder: TransformerDecoder) -> None:
        """Test proper initialization of TransformerDecoder."""
        assert transformer_decoder.num_layers == 2
        assert transformer_decoder.num_feature_levels == 3
        assert len(transformer_decoder.layers) == 2
        assert transformer_decoder.num_feature_levels == 3
        assert transformer_decoder.initial_queries.shape == (1, 100, 256)
        assert transformer_decoder.pos_query_embed.shape == (1, 100, 256)

    def test_forward_shape(
        self,
        transformer_decoder: TransformerDecoder,
        transformer_inputs: dict[str, torch.Tensor],
    ) -> None:
        """Test output shapes are correct."""
        output = transformer_decoder(**transformer_inputs)

        batch_size = 2
        num_queries = 100
        embedding_dim = 256
        mask_height, mask_width = 32, 32

        assert "query_features" in output
        assert "masks" in output
        assert "auxiliary_masks" in output

        assert output["query_features"].shape == (
            batch_size,
            num_queries,
            embedding_dim,
        )
        assert output["masks"].shape == (
            batch_size,
            num_queries,
            mask_height,
            mask_width,
        )

        # Auxiliary masks: (layers, feature_levels, batch, queries, h, w)
        assert output["auxiliary_masks"].shape == (
            2,
            3,
            batch_size,
            num_queries,
            mask_height,
            mask_width,
        )

    def test_forward_no_auxiliary_masks(
        self,
        transformer_decoder: TransformerDecoder,
        transformer_inputs: dict[str, torch.Tensor],
    ) -> None:
        """Test forward pass without auxiliary masks."""
        output = transformer_decoder(**transformer_inputs, return_auxiliary_masks=False)

        assert "auxiliary_masks" in output
        assert output["auxiliary_masks"].numel() == 0  # Empty tensor

    def test_gradient_flow(
        self,
        transformer_decoder: TransformerDecoder,
        transformer_inputs: dict[str, torch.Tensor],
    ) -> None:
        """Test gradients flow through the decoder."""
        # Create fresh copies with gradients enabled
        inputs_copy = {
            "image_features_list": [
                feat.clone().requires_grad_(True)
                for feat in transformer_inputs["image_features_list"]
            ],
            "mask_features": transformer_inputs["mask_features"]
            .clone()
            .requires_grad_(True),
        }

        output = transformer_decoder(**inputs_copy)
        loss = output["query_features"].sum() + output["masks"].sum()
        loss.backward()

        # Check gradients exist for input tensors
        for feat in inputs_copy["image_features_list"]:
            assert feat.grad is not None
            assert not torch.allclose(feat.grad, torch.zeros_like(feat.grad))

        mask_features = inputs_copy["mask_features"]
        assert isinstance(mask_features, torch.Tensor)
        assert mask_features.grad is not None

    def test_different_batch_sizes(self, test_generator: torch.Generator) -> None:
        """Test with different batch sizes."""
        decoder = TransformerDecoder(
            num_layers=1,
            embedding_dim=128,
            num_query=50,
            num_head=4,
            hidden_dim=512,
            num_feature_levels=2,
            input_size=(32, 32),
            output_divisors=(4, 2),
        )

        for batch_size in [1, 3, 5]:
            inputs = {
                "image_features_list": [
                    torch.randn(batch_size, 128, 8, 8, generator=test_generator),
                    torch.randn(batch_size, 128, 16, 16, generator=test_generator),
                ],
                "mask_features": torch.randn(
                    batch_size, 128, 16, 16, generator=test_generator
                ),
            }

            output = decoder(**inputs)
            assert output["query_features"].shape == (batch_size, 50, 128)
            assert output["masks"].shape == (batch_size, 50, 16, 16)

    def test_parameter_initialization(self) -> None:
        """Test parameter initialization is reasonable."""
        decoder = TransformerDecoder(
            num_layers=1,
            embedding_dim=256,
            num_query=100,
            num_head=8,
            hidden_dim=1024,
        )

        # Check parameters are initialized with reasonable values
        assert not torch.allclose(
            decoder.initial_queries, torch.zeros_like(decoder.initial_queries)
        )
        assert not torch.allclose(
            decoder.pos_query_embed, torch.zeros_like(decoder.pos_query_embed)
        )

        # Check Xavier initialization bounds (roughly)
        bound = (6.0 / (1 + 256)) ** 0.5  # Xavier uniform bound
        assert torch.all(
            torch.abs(decoder.initial_queries) <= bound * 2
        )  # Allow some variance

    def test_positional_embeddings_buffered(
        self, transformer_decoder: TransformerDecoder
    ) -> None:
        """Test positional embeddings are properly registered as buffers."""
        buffers = dict(transformer_decoder.named_buffers())

        assert "ape_0" in buffers
        assert "ape_1" in buffers
        assert "ape_2" in buffers

        # Check shapes match expected resolutions
        assert buffers["ape_0"].shape == (1, 64, 256)  # 8x8 flattened
        assert buffers["ape_1"].shape == (1, 256, 256)  # 16x16 flattened
        assert buffers["ape_2"].shape == (1, 1024, 256)  # 32x32 flattened

    def test_torch_compile_compatibility(
        self,
        transformer_decoder: TransformerDecoder,
        transformer_inputs: dict[str, torch.Tensor],
    ) -> None:
        """Test TransformerDecoder is compatible with torch.compile."""
        compiled_decoder = torch.compile(transformer_decoder, fullgraph=True)

        # Run compiled version
        compiled_output = compiled_decoder(**transformer_inputs)

        # Run original version
        original_output = transformer_decoder(**transformer_inputs)

        # Outputs should be close (torch.compile can have numerical differences)
        torch.testing.assert_close(
            compiled_output["query_features"],
            original_output["query_features"],
            atol=1e-3,
            rtol=1e-3,
        )
        torch.testing.assert_close(
            compiled_output["masks"], original_output["masks"], atol=1e-3, rtol=1e-3
        )

    @pytest.mark.skipif(
        not hasattr(torch, "export") or torch.__version__ < "2.1",
        reason="torch.export requires PyTorch 2.1+",
    )
    def test_torch_export_compatibility(
        self,
        transformer_inputs: dict[str, torch.Tensor],
    ) -> None:
        """Test TransformerDecoder is compatible with torch.export."""
        # Use smaller decoder for export (reduces complexity)
        decoder = TransformerDecoder(
            num_layers=1,
            embedding_dim=128,
            num_query=50,
            num_head=4,
            hidden_dim=512,
            num_feature_levels=2,
            input_size=(32, 32),
            output_divisors=(4, 2),
        )

        # Create compatible inputs
        export_inputs = {
            "image_features_list": [
                torch.randn(1, 128, 8, 8),
                torch.randn(1, 128, 16, 16),
            ],
            "mask_features": torch.randn(1, 128, 16, 16),
        }

        # Test export (this will raise if incompatible)
        try:
            exported_program = torch.export.export(decoder, (), export_inputs)
            assert exported_program is not None
        except Exception as e:
            logger.error("torch.export failed: %s", str(e))
            pytest.fail(f"torch.export failed - may be environment specific: {e}")

    @pytest.mark.benchmark
    def test_compiled_vs_eager_benchmark(
        self,
        transformer_decoder: TransformerDecoder,
        transformer_inputs: dict[str, torch.Tensor],
    ) -> None:
        """Benchmark compiled vs eager execution."""
        compiled_decoder = torch.compile(transformer_decoder)
        assert isinstance(compiled_decoder, torch.nn.Module)

        num_runs = 10

        # Warmup both versions
        logger.info("Warming up both eager and compiled decoders...")
        for _ in range(3):
            _ = transformer_decoder(**transformer_inputs)
            _ = compiled_decoder(**transformer_inputs)

        # Benchmark eager execution
        logger.info("Benchmarking eager execution...")
        eager_times = _benchmark_execution(
            transformer_decoder, transformer_inputs, num_runs
        )

        # Benchmark compiled execution
        logger.info("Benchmarking compiled execution...")
        compiled_times = _benchmark_execution(
            compiled_decoder, transformer_inputs, num_runs
        )

        # Calculate statistics
        eager_stats = _calculate_stats(eager_times)
        compiled_stats = _calculate_stats(compiled_times)
        speedup = (
            eager_stats["avg"] / compiled_stats["avg"]
            if compiled_stats["avg"] > 0
            else 0
        )

        # Log results
        logger.info("Compilation benchmark results:")
        logger.info(
            "  Eager - avg: %.2fms, min: %.2fms, max: %.2fms",
            eager_stats["avg"] * 1000,
            eager_stats["min"] * 1000,
            eager_stats["max"] * 1000,
        )
        logger.info(
            "  Compiled - avg: %.2fms, min: %.2fms, max: %.2fms",
            compiled_stats["avg"] * 1000,
            compiled_stats["min"] * 1000,
            compiled_stats["max"] * 1000,
        )
        logger.info("  Speedup: %.2fx", speedup)

        # Performance assertions
        assert eager_stats["avg"] < 1.0, (
            f"Eager execution too slow: {eager_stats['avg']:.3f}s"
        )
        assert compiled_stats["avg"] < 1.0, (
            f"Compiled execution too slow: {compiled_stats['avg']:.3f}s"
        )
        assert speedup > 1.1, f"Compilation should improve performance: {speedup:.2f}x"
