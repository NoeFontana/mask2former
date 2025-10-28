import logging

import pytest
import torch

from mask2former.modeling.attn import CrossAttention
from tests.benchmark import benchmark_module

logger = logging.getLogger(__name__)


@pytest.fixture(scope="class")
def standard_cross_attention() -> CrossAttention:
    """Fixture providing a standard CrossAttention for testing."""
    return CrossAttention(embedding_dim=256, num_head=8)


@pytest.fixture(scope="class")
def small_cross_attention() -> CrossAttention:
    """Fixture providing a small CrossAttention for quick tests."""
    return CrossAttention(embedding_dim=64, num_head=2)


class TestCrossAttention:
    """Tests for the CrossAttention class."""

    def test_initialization(self, standard_cross_attention: CrossAttention) -> None:
        """Test proper initialization of CrossAttention."""
        attention = standard_cross_attention

        # Check multi-head attributes
        assert attention.num_head == 8
        assert attention.head_embedding_dim == 32  # 256 // 8

        # Check that projectors are initialized correctly
        assert isinstance(attention.query_projector, torch.nn.Linear)
        assert isinstance(attention.key_projector, torch.nn.Linear)
        assert isinstance(attention.value_projector, torch.nn.Linear)
        assert isinstance(attention.out_proj, torch.nn.Linear)

        # Check dimensions
        assert attention.query_projector.in_features == 256
        assert attention.query_projector.out_features == 256
        assert attention.key_projector.in_features == 256
        assert attention.key_projector.out_features == 256
        assert attention.value_projector.in_features == 256
        assert attention.value_projector.out_features == 256
        assert attention.out_proj.in_features == 256
        assert attention.out_proj.out_features == 256

    def test_forward_pass_shape(self, standard_cross_attention: CrossAttention) -> None:
        """Test forward pass produces correct output shape."""
        attention = standard_cross_attention
        batch_size, num_queries, embedding_dim = 2, 100, 256
        height, width = 32, 32

        # Create input tensors
        query_features = torch.randn(batch_size, num_queries, embedding_dim)
        pos_query_embeddings = torch.randn(batch_size, num_queries, embedding_dim)
        image_features = torch.randn(batch_size, embedding_dim, height, width)
        mask = torch.randint(0, 2, (batch_size, num_queries, height, width)).bool()
        pos_image_embeddings = torch.randn(batch_size, height * width, embedding_dim)

        # Forward pass
        output = attention(
            query_features,
            image_features,
            mask,
            pos_query_embeddings,
            pos_image_embeddings,
        )

        # Check output shape matches input query shape
        assert output.shape == (batch_size, num_queries, embedding_dim)
        assert output.dtype == torch.float32

    def test_residual_connection(
        self, standard_cross_attention: CrossAttention
    ) -> None:
        """Test that residual connection is applied correctly."""
        attention = standard_cross_attention
        batch_size, num_queries, embedding_dim = 1, 10, 256
        height, width = 8, 8

        query_features = torch.randn(batch_size, num_queries, embedding_dim)
        pos_query_embeddings = torch.randn(batch_size, num_queries, embedding_dim)
        image_features = torch.randn(batch_size, embedding_dim, height, width)
        mask = torch.ones(batch_size, num_queries, height, width).bool()
        pos_image_embeddings = torch.randn(batch_size, height * width, embedding_dim)

        output = attention(
            query_features,
            image_features,
            mask,
            pos_query_embeddings,
            pos_image_embeddings,
        )

        # The output should not be equal to input due to attention computation
        # but should have the same shape due to residual connection
        assert output.shape == query_features.shape
        assert not torch.allclose(output, query_features, rtol=1e-3)

    def test_mask_effect(self, standard_cross_attention: CrossAttention) -> None:
        """Test that different masks produce different outputs."""
        attention = standard_cross_attention
        batch_size, num_queries, embedding_dim = 1, 5, 256
        height, width = 4, 4

        query_features = torch.randn(batch_size, num_queries, embedding_dim)
        pos_query_embeddings = torch.randn(batch_size, num_queries, embedding_dim)
        image_features = torch.randn(batch_size, embedding_dim, height, width)
        pos_image_embeddings = torch.randn(batch_size, height * width, embedding_dim)

        # Create two different masks
        mask_all_true = torch.ones(batch_size, num_queries, height, width).bool()
        mask_partial = torch.zeros(batch_size, num_queries, height, width).bool()
        mask_partial[:, :, : height // 2, :] = True  # Only attend to top half

        output_full = attention(
            query_features,
            image_features,
            mask_all_true,
            pos_query_embeddings,
            pos_image_embeddings,
        )
        output_partial = attention(
            query_features,
            image_features,
            mask_partial,
            pos_query_embeddings,
            pos_image_embeddings,
        )

        # Different masks should produce different outputs
        assert not torch.allclose(output_full, output_partial, rtol=1e-3)

    def test_mask_all_false(self, standard_cross_attention: CrossAttention) -> None:
        """Test behavior when mask is all False."""
        attention = standard_cross_attention
        batch_size, num_queries, embedding_dim = 1, 3, 256
        height, width = 4, 4

        query_features = torch.randn(batch_size, num_queries, embedding_dim)
        pos_query_embeddings = torch.randn(batch_size, num_queries, embedding_dim)
        image_features = torch.randn(batch_size, embedding_dim, height, width)
        mask_all_false = torch.zeros(batch_size, num_queries, height, width).bool()
        pos_image_embeddings = torch.randn(batch_size, height * width, embedding_dim)

        output = attention(
            query_features,
            image_features,
            mask_all_false,
            pos_query_embeddings,
            pos_image_embeddings,
        )

        # Output should still have correct shape
        assert output.shape == query_features.shape
        # With all-false mask, attention should be masked out, but residual keeps
        # original features. The exact behavior depends on PyTorch's
        # scaled_dot_product_attention implementation

    def test_gradient_flow(self, standard_cross_attention: CrossAttention) -> None:
        """Test that gradients flow properly through the attention module."""
        attention = standard_cross_attention
        batch_size, num_queries, embedding_dim = 1, 10, 256
        height, width = 8, 8

        query_features = torch.randn(
            batch_size, num_queries, embedding_dim, requires_grad=True
        )
        image_features = torch.randn(
            batch_size, embedding_dim, height, width, requires_grad=True
        )
        mask = torch.randint(0, 2, (batch_size, num_queries, height, width)).bool()
        pos_query_embeddings = torch.randn(batch_size, num_queries, embedding_dim)
        pos_image_embeddings = torch.randn(batch_size, height * width, embedding_dim)

        output = attention(
            query_features,
            image_features,
            mask,
            pos_query_embeddings,
            pos_image_embeddings,
        )
        loss = output.sum()
        loss.backward()

        # Check that gradients are computed for inputs
        assert query_features.grad is not None
        assert image_features.grad is not None

        # Check that gradients are computed for parameters
        assert attention.query_projector.weight.grad is not None
        assert attention.key_projector.weight.grad is not None
        assert attention.value_projector.weight.grad is not None

    @pytest.mark.parametrize(
        "embedding_dim,num_head", [(64, 4), (128, 8), (256, 8), (512, 16)]
    )
    def test_different_embedding_dimensions(
        self, embedding_dim: int, num_head: int
    ) -> None:
        """Test with different embedding dimensions and head counts."""
        attention = CrossAttention(embedding_dim=embedding_dim, num_head=num_head)
        batch_size, num_queries = 2, 20
        height, width = 16, 16

        query_features = torch.randn(batch_size, num_queries, embedding_dim)
        pos_query_embeddings = torch.randn(batch_size, num_queries, embedding_dim)
        image_features = torch.randn(batch_size, embedding_dim, height, width)
        mask = torch.randint(0, 2, (batch_size, num_queries, height, width)).bool()
        pos_image_embeddings = torch.randn(batch_size, height * width, embedding_dim)

        output = attention(
            query_features,
            image_features,
            mask,
            pos_query_embeddings,
            pos_image_embeddings,
        )

        assert output.shape == (batch_size, num_queries, embedding_dim)
        assert output.dtype == torch.float32

    @pytest.mark.parametrize("height,width", [(8, 8), (16, 32), (64, 64)])
    def test_different_spatial_sizes(
        self, small_cross_attention: CrossAttention, height: int, width: int
    ) -> None:
        """Test with different spatial dimensions."""
        attention = small_cross_attention
        batch_size, num_queries, embedding_dim = 1, 10, 64

        query_features = torch.randn(batch_size, num_queries, embedding_dim)
        pos_query_embeddings = torch.randn(batch_size, num_queries, embedding_dim)
        image_features = torch.randn(batch_size, embedding_dim, height, width)
        mask = torch.randint(0, 2, (batch_size, num_queries, height, width)).bool()
        pos_image_embeddings = torch.randn(batch_size, height * width, embedding_dim)

        output = attention(
            query_features,
            image_features,
            mask,
            pos_query_embeddings,
            pos_image_embeddings,
        )

        assert output.shape == (batch_size, num_queries, embedding_dim)

    @pytest.mark.parametrize("batch_size,num_queries", [(1, 50), (3, 100), (8, 200)])
    def test_different_batch_and_query_sizes(
        self,
        small_cross_attention: CrossAttention,
        batch_size: int,
        num_queries: int,
    ) -> None:
        """Test with different batch sizes and number of queries."""
        attention = small_cross_attention
        embedding_dim = 64
        height, width = 16, 16

        query_features = torch.randn(batch_size, num_queries, embedding_dim)
        pos_query_embeddings = torch.randn(batch_size, num_queries, embedding_dim)
        image_features = torch.randn(batch_size, embedding_dim, height, width)
        mask = torch.randint(0, 2, (batch_size, num_queries, height, width)).bool()
        pos_image_embeddings = torch.randn(batch_size, height * width, embedding_dim)

        output = attention(
            query_features,
            image_features,
            mask,
            pos_query_embeddings,
            pos_image_embeddings,
        )

        assert output.shape == (batch_size, num_queries, embedding_dim)
        assert output.dtype == torch.float32

    def test_attention_deterministic(
        self, standard_cross_attention: CrossAttention
    ) -> None:
        """Test that attention produces deterministic results with same inputs."""
        attention = standard_cross_attention
        batch_size, num_queries, embedding_dim = 1, 5, 256
        height, width = 8, 8

        # Set seeds for reproducibility
        torch.manual_seed(42)
        query_features = torch.randn(batch_size, num_queries, embedding_dim)
        pos_query_embeddings = torch.randn(batch_size, num_queries, embedding_dim)
        image_features = torch.randn(batch_size, embedding_dim, height, width)
        mask = torch.randint(0, 2, (batch_size, num_queries, height, width)).bool()
        pos_image_embeddings = torch.randn(batch_size, height * width, embedding_dim)

        # Run twice with same inputs
        torch.manual_seed(42)
        output1 = attention(
            query_features,
            image_features,
            mask,
            pos_query_embeddings,
            pos_image_embeddings,
        )

        torch.manual_seed(42)
        output2 = attention(
            query_features,
            image_features,
            mask,
            pos_query_embeddings,
            pos_image_embeddings,
        )

        # Outputs should be identical
        assert torch.allclose(output1, output2, rtol=1e-6, atol=1e-8)

    def test_cross_attention_compilation(
        self, standard_cross_attention: CrossAttention
    ) -> None:
        """Test that CrossAttention compiles without graph breaks."""
        attention = standard_cross_attention
        batch_size, num_queries, embedding_dim = 2, 50, 256
        height, width = 16, 16

        query_features = torch.randn(batch_size, num_queries, embedding_dim)
        pos_query_embeddings = torch.randn(batch_size, num_queries, embedding_dim)
        image_features = torch.randn(batch_size, embedding_dim, height, width)
        mask = torch.randint(0, 2, (batch_size, num_queries, height, width)).bool()
        pos_image_embeddings = torch.randn(batch_size, height * width, embedding_dim)

        # Compile the model
        compiled_attention = torch.compile(attention, fullgraph=True)

        # Run compiled version
        compiled_output = compiled_attention(
            query_features,
            image_features,
            mask,
            pos_query_embeddings,
            pos_image_embeddings,
        )

        # Run original version for comparison
        original_output = attention(
            query_features,
            image_features,
            mask,
            pos_query_embeddings,
            pos_image_embeddings,
        )

        # Check outputs are equivalent
        assert compiled_output.shape == original_output.shape
        assert torch.allclose(compiled_output, original_output, rtol=1e-4)

    def test_image_features_flattening(
        self, standard_cross_attention: CrossAttention
    ) -> None:
        """Test that image features are correctly flattened and permuted."""
        attention = standard_cross_attention
        batch_size, num_queries, embedding_dim = 1, 5, 256
        height, width = 4, 8

        query_features = torch.randn(batch_size, num_queries, embedding_dim)
        pos_query_embeddings = torch.randn(batch_size, num_queries, embedding_dim)
        image_features = torch.randn(batch_size, embedding_dim, height, width)
        pos_image_embeddings = torch.randn(batch_size, height * width, embedding_dim)
        mask = torch.ones(batch_size, num_queries, height, width).bool()

        # Manually check the transformation
        expected_flattened = image_features.flatten(2).permute(0, 2, 1)
        assert expected_flattened.shape == (batch_size, height * width, embedding_dim)

        # Should not raise any errors
        output = attention(
            query_features,
            image_features,
            mask,
            pos_query_embeddings,
            pos_image_embeddings,
        )
        assert output.shape == (batch_size, num_queries, embedding_dim)

    def test_mask_flattening(self, standard_cross_attention: CrossAttention) -> None:
        """Test that mask is correctly flattened."""
        attention = standard_cross_attention
        batch_size, num_queries, embedding_dim = 1, 3, 256
        height, width = 4, 6

        query_features = torch.randn(batch_size, num_queries, embedding_dim)
        pos_query_embeddings = torch.randn(batch_size, num_queries, embedding_dim)
        image_features = torch.randn(batch_size, embedding_dim, height, width)
        mask = torch.randint(0, 2, (batch_size, num_queries, height, width)).bool()
        pos_image_embeddings = torch.randn(batch_size, height * width, embedding_dim)

        # Manually check mask transformation
        expected_mask_flat = mask.flatten(2)
        assert expected_mask_flat.shape == (batch_size, num_queries, height * width)

        # Should not raise any errors
        output = attention(
            query_features,
            image_features,
            mask,
            pos_query_embeddings,
            pos_image_embeddings,
        )
        assert output.shape == (batch_size, num_queries, embedding_dim)

    def test_multi_head_validation(self) -> None:
        """Test multi-head attention parameter validation."""
        # Test valid configurations
        attention_valid = CrossAttention(embedding_dim=256, num_head=8)
        assert attention_valid.num_head == 8
        assert attention_valid.head_embedding_dim == 32

        # Test invalid configuration (not divisible)
        with pytest.raises(
            ValueError, match="embedding_dim must be divisible by num_head"
        ):
            CrossAttention(embedding_dim=257, num_head=8)

    def test_single_head_attention(self) -> None:
        """Test single-head attention (should use Identity for out_proj)."""
        attention = CrossAttention(embedding_dim=64, num_head=1)

        assert attention.num_head == 1
        assert attention.head_embedding_dim == 64
        assert isinstance(attention.out_proj, torch.nn.Identity)

        batch_size, num_queries = 2, 10
        height, width = 8, 8

        query_features = torch.randn(batch_size, num_queries, 64)
        pos_query_embeddings = torch.randn(batch_size, num_queries, 64)
        image_features = torch.randn(batch_size, 64, height, width)
        mask = torch.randint(0, 2, (batch_size, num_queries, height, width)).bool()
        pos_image_embeddings = torch.randn(batch_size, height * width, 64)

        output = attention(
            query_features,
            image_features,
            mask,
            pos_query_embeddings,
            pos_image_embeddings,
        )
        assert output.shape == (batch_size, num_queries, 64)

    @pytest.mark.parametrize("num_head", [1, 2, 4, 8, 16])
    def test_different_head_counts(self, num_head: int) -> None:
        """Test with different numbers of attention heads."""
        embedding_dim = 128
        attention = CrossAttention(embedding_dim=embedding_dim, num_head=num_head)

        batch_size, num_queries = 2, 20
        height, width = 16, 16

        query_features = torch.randn(batch_size, num_queries, embedding_dim)
        pos_query_embeddings = torch.randn(batch_size, num_queries, embedding_dim)
        image_features = torch.randn(batch_size, embedding_dim, height, width)
        mask = torch.randint(0, 2, (batch_size, num_queries, height, width)).bool()
        pos_image_embeddings = torch.randn(batch_size, height * width, embedding_dim)

        output = attention(
            query_features,
            image_features,
            mask,
            pos_query_embeddings,
            pos_image_embeddings,
        )

        assert output.shape == (batch_size, num_queries, embedding_dim)
        assert attention.head_embedding_dim == embedding_dim // num_head

    def test_head_reshaping(self) -> None:
        """Test that tensors are correctly reshaped for multi-head attention."""
        attention = CrossAttention(embedding_dim=256, num_head=8)

        batch_size, num_queries = 1, 5
        height, width = 4, 4

        query_features = torch.randn(batch_size, num_queries, 256)
        pos_query_embeddings = torch.randn(batch_size, num_queries, 256)
        image_features = torch.randn(batch_size, 256, height, width)
        pos_image_embeddings = torch.randn(batch_size, height * width, 256)
        mask = torch.ones(batch_size, num_queries, height, width).bool()

        # Test that forward pass completes successfully with correct shapes
        output = attention(
            query_features,
            image_features,
            mask,
            pos_query_embeddings,
            pos_image_embeddings,
        )
        assert output.shape == (batch_size, num_queries, 256)

    def test_compile_validation(self, standard_cross_attention: CrossAttention) -> None:
        """Test that compilation works without errors and produces valid results."""
        attention = standard_cross_attention
        batch_size, num_queries, embedding_dim = 2, 50, 256
        height, width = 16, 16

        query_features = torch.randn(batch_size, num_queries, embedding_dim)
        pos_query_embeddings = torch.randn(batch_size, num_queries, embedding_dim)
        image_features = torch.randn(batch_size, embedding_dim, height, width)
        mask = torch.randint(0, 2, (batch_size, num_queries, height, width)).bool()
        pos_image_embeddings = torch.randn(batch_size, height * width, embedding_dim)

        # Test compilation with different modes
        try:
            # Test fullgraph compilation
            compiled_attention_full = torch.compile(attention, fullgraph=True)
            output_full = compiled_attention_full(
                query_features,
                image_features,
                mask,
                pos_query_embeddings,
                pos_image_embeddings,
            )
            assert output_full.shape == (batch_size, num_queries, embedding_dim)

            # Test default compilation
            compiled_attention_default = torch.compile(attention)
            output_default = compiled_attention_default(
                query_features,
                image_features,
                mask,
                pos_query_embeddings,
                pos_image_embeddings,
            )
            assert output_default.shape == (batch_size, num_queries, embedding_dim)

            # Test that compiled versions produce consistent results
            original_output = attention(
                query_features,
                image_features,
                mask,
                pos_query_embeddings,
                pos_image_embeddings,
            )
            assert torch.allclose(output_full, original_output, rtol=1e-4, atol=1e-5)
            assert torch.allclose(output_default, original_output, rtol=1e-4, atol=1e-5)

        except Exception as e:
            pytest.fail(f"Compilation failed with error: {e}")

    @pytest.mark.benchmark
    @pytest.mark.parametrize("feature_size", [32, 64, 128])
    @torch.inference_mode()
    def test_performance_benchmark(
        self, standard_cross_attention: CrossAttention, feature_size: int
    ) -> None:
        """Benchmark test for CrossAttention performance.

        This test measures the performance of the CrossAttention module
        with realistic input sizes. It's marked as 'benchmark' to be
        disabled by default in CI.
        """
        attention = standard_cross_attention
        batch_size, num_queries, embedding_dim = 8, 200, 256
        height, width = feature_size, feature_size

        query_features = torch.randn(batch_size, num_queries, embedding_dim)
        pos_query_embeddings = torch.randn(batch_size, num_queries, embedding_dim)
        image_features = torch.randn(batch_size, embedding_dim, height, width)
        mask = torch.randint(0, 2, (batch_size, num_queries, height, width)).bool()
        pos_image_embeddings = torch.randn(batch_size, height * width, embedding_dim)

        # Benchmark uncompiled version
        uncompiled_results = benchmark_module(
            attention,
            query_features,
            image_features,
            mask,
            pos_query_embeddings,
            pos_image_embeddings,
        )

        # Basic sanity checks
        assert uncompiled_results["output_shape"] == (
            batch_size,
            num_queries,
            embedding_dim,
        )
        assert uncompiled_results["avg_time_ms"] > 0

        logger.info(
            "\nCrossAttention Benchmark Results\n"
            + ("=" * 50 + "\n")
            + f"Device: {uncompiled_results['device']}\n"
            f"Input shapes - Queries: {query_features.shape}, "
            f"Images: {image_features.shape}, Mask: {mask.shape}\n"
            f"Batch size: {batch_size}, Num queries: {num_queries}\n"
            f"Average forward pass time: {uncompiled_results['avg_time_ms']:.3f} ms\n"
            f"Throughput: {uncompiled_results['throughput']:.2f} samples/sec"
        )

        # Benchmark compiled version
        compiled_attention = torch.compile(attention, fullgraph=True)
        compiled_results = benchmark_module(
            compiled_attention,
            query_features,
            image_features,
            mask,
            pos_query_embeddings,
            pos_image_embeddings,
        )

        # Basic sanity checks
        assert compiled_results["output_shape"] == (
            batch_size,
            num_queries,
            embedding_dim,
        )
        assert compiled_results["avg_time_ms"] > 0

        speedup = uncompiled_results["avg_time_ms"] / compiled_results["avg_time_ms"]
        logger.info(
            "\nCompiled Version Results:\n"
            f"Compiled forward pass time: {compiled_results['avg_time_ms']:.3f} ms\n"
            f"Compiled throughput: {compiled_results['throughput']:.2f} samples/sec\n"
            f"Compilation speedup: {speedup:.2f}x\n" + "=" * 50
        )

        if compiled_results["avg_time_ms"] > uncompiled_results["avg_time_ms"]:
            logger.warning(
                "Compiled version was not faster than uncompiled version.\n"
                f"Uncompiled: {uncompiled_results['avg_time_ms']:.3f} ms\n"
                f"Compiled: {compiled_results['avg_time_ms']:.3f} ms"
            )
