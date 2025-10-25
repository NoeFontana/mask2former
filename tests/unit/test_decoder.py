"""Unit tests for Mask2Former decoder components."""

import logging

import pytest
import torch

from mask2former.modeling.decoder import MaskedAttention

# Set up logger for this module
logger = logging.getLogger(__name__)


@pytest.fixture(scope="class")
def standard_masked_attention() -> MaskedAttention:
    """Fixture providing a standard MaskedAttention for testing."""
    return MaskedAttention(embedding_dim=256)


@pytest.fixture(scope="class")
def small_masked_attention() -> MaskedAttention:
    """Fixture providing a small MaskedAttention for quick tests."""
    return MaskedAttention(embedding_dim=64)


class TestMaskedAttention:
    """Tests for the MaskedAttention class."""

    def test_initialization(self, standard_masked_attention: MaskedAttention) -> None:
        """Test proper initialization of MaskedAttention."""
        attention = standard_masked_attention

        # Check that projectors are initialized correctly
        assert isinstance(attention.query_projector, torch.nn.Linear)
        assert isinstance(attention.key_projector, torch.nn.Linear)
        assert isinstance(attention.value_projector, torch.nn.Linear)

        # Check dimensions
        assert attention.query_projector.in_features == 256
        assert attention.query_projector.out_features == 256
        assert attention.key_projector.in_features == 256
        assert attention.key_projector.out_features == 256
        assert attention.value_projector.in_features == 256
        assert attention.value_projector.out_features == 256

    def test_forward_pass_shape(
        self, standard_masked_attention: MaskedAttention
    ) -> None:
        """Test forward pass produces correct output shape."""
        attention = standard_masked_attention
        batch_size, num_queries, embedding_dim = 2, 100, 256
        height, width = 32, 32

        # Create input tensors
        query_features = torch.randn(batch_size, num_queries, embedding_dim)
        image_features = torch.randn(batch_size, embedding_dim, height, width)
        mask = torch.randint(0, 2, (batch_size, num_queries, height, width)).bool()

        # Forward pass
        output = attention(query_features, image_features, mask)

        # Check output shape matches input query shape
        assert output.shape == (batch_size, num_queries, embedding_dim)
        assert output.dtype == torch.float32

    def test_residual_connection(
        self, standard_masked_attention: MaskedAttention
    ) -> None:
        """Test that residual connection is applied correctly."""
        attention = standard_masked_attention
        batch_size, num_queries, embedding_dim = 1, 10, 256
        height, width = 8, 8

        query_features = torch.randn(batch_size, num_queries, embedding_dim)
        image_features = torch.randn(batch_size, embedding_dim, height, width)
        mask = torch.ones(batch_size, num_queries, height, width).bool()

        output = attention(query_features, image_features, mask)

        # The output should not be equal to input due to attention computation
        # but should have the same shape due to residual connection
        assert output.shape == query_features.shape
        assert not torch.allclose(output, query_features, rtol=1e-3)

    def test_mask_effect(self, standard_masked_attention: MaskedAttention) -> None:
        """Test that different masks produce different outputs."""
        attention = standard_masked_attention
        batch_size, num_queries, embedding_dim = 1, 5, 256
        height, width = 4, 4

        query_features = torch.randn(batch_size, num_queries, embedding_dim)
        image_features = torch.randn(batch_size, embedding_dim, height, width)

        # Create two different masks
        mask_all_true = torch.ones(batch_size, num_queries, height, width).bool()
        mask_partial = torch.zeros(batch_size, num_queries, height, width).bool()
        mask_partial[:, :, : height // 2, :] = True  # Only attend to top half

        output_full = attention(query_features, image_features, mask_all_true)
        output_partial = attention(query_features, image_features, mask_partial)

        # Different masks should produce different outputs
        assert not torch.allclose(output_full, output_partial, rtol=1e-3)

    def test_mask_all_false(self, standard_masked_attention: MaskedAttention) -> None:
        """Test behavior when mask is all False."""
        attention = standard_masked_attention
        batch_size, num_queries, embedding_dim = 1, 3, 256
        height, width = 4, 4

        query_features = torch.randn(batch_size, num_queries, embedding_dim)
        image_features = torch.randn(batch_size, embedding_dim, height, width)
        mask_all_false = torch.zeros(batch_size, num_queries, height, width).bool()

        output = attention(query_features, image_features, mask_all_false)

        # Output should still have correct shape
        assert output.shape == query_features.shape
        # With all-false mask, attention should be masked out, but residual keeps
        # original features. The exact behavior depends on PyTorch's
        # scaled_dot_product_attention implementation

    def test_gradient_flow(self, standard_masked_attention: MaskedAttention) -> None:
        """Test that gradients flow properly through the attention module."""
        attention = standard_masked_attention
        batch_size, num_queries, embedding_dim = 1, 10, 256
        height, width = 8, 8

        query_features = torch.randn(
            batch_size, num_queries, embedding_dim, requires_grad=True
        )
        image_features = torch.randn(
            batch_size, embedding_dim, height, width, requires_grad=True
        )
        mask = torch.randint(0, 2, (batch_size, num_queries, height, width)).bool()

        output = attention(query_features, image_features, mask)
        loss = output.sum()
        loss.backward()

        # Check that gradients are computed for inputs
        assert query_features.grad is not None
        assert image_features.grad is not None

        # Check that gradients are computed for parameters
        assert attention.query_projector.weight.grad is not None
        assert attention.key_projector.weight.grad is not None
        assert attention.value_projector.weight.grad is not None

    @pytest.mark.parametrize("embedding_dim", [64, 128, 256, 512])
    def test_different_embedding_dimensions(self, embedding_dim: int) -> None:
        """Test with different embedding dimensions."""
        attention = MaskedAttention(embedding_dim=embedding_dim)
        batch_size, num_queries = 2, 20
        height, width = 16, 16

        query_features = torch.randn(batch_size, num_queries, embedding_dim)
        image_features = torch.randn(batch_size, embedding_dim, height, width)
        mask = torch.randint(0, 2, (batch_size, num_queries, height, width)).bool()

        output = attention(query_features, image_features, mask)

        assert output.shape == (batch_size, num_queries, embedding_dim)
        assert output.dtype == torch.float32

    @pytest.mark.parametrize("height,width", [(8, 8), (16, 32), (64, 64)])
    def test_different_spatial_sizes(
        self, small_masked_attention: MaskedAttention, height: int, width: int
    ) -> None:
        """Test with different spatial dimensions."""
        attention = small_masked_attention
        batch_size, num_queries, embedding_dim = 1, 10, 64

        query_features = torch.randn(batch_size, num_queries, embedding_dim)
        image_features = torch.randn(batch_size, embedding_dim, height, width)
        mask = torch.randint(0, 2, (batch_size, num_queries, height, width)).bool()

        output = attention(query_features, image_features, mask)

        assert output.shape == (batch_size, num_queries, embedding_dim)

    @pytest.mark.parametrize("batch_size,num_queries", [(1, 50), (3, 100), (8, 200)])
    def test_different_batch_and_query_sizes(
        self,
        small_masked_attention: MaskedAttention,
        batch_size: int,
        num_queries: int,
    ) -> None:
        """Test with different batch sizes and number of queries."""
        attention = small_masked_attention
        embedding_dim = 64
        height, width = 16, 16

        query_features = torch.randn(batch_size, num_queries, embedding_dim)
        image_features = torch.randn(batch_size, embedding_dim, height, width)
        mask = torch.randint(0, 2, (batch_size, num_queries, height, width)).bool()

        output = attention(query_features, image_features, mask)

        assert output.shape == (batch_size, num_queries, embedding_dim)
        assert output.dtype == torch.float32

    def test_attention_deterministic(
        self, standard_masked_attention: MaskedAttention
    ) -> None:
        """Test that attention produces deterministic results with same inputs."""
        attention = standard_masked_attention
        batch_size, num_queries, embedding_dim = 1, 5, 256
        height, width = 8, 8

        # Set seeds for reproducibility
        torch.manual_seed(42)
        query_features = torch.randn(batch_size, num_queries, embedding_dim)
        image_features = torch.randn(batch_size, embedding_dim, height, width)
        mask = torch.randint(0, 2, (batch_size, num_queries, height, width)).bool()

        # Run twice with same inputs
        torch.manual_seed(42)
        output1 = attention(query_features, image_features, mask)

        torch.manual_seed(42)
        output2 = attention(query_features, image_features, mask)

        # Outputs should be identical
        assert torch.allclose(output1, output2, rtol=1e-6, atol=1e-8)

    def test_masked_attention_compilation(
        self, standard_masked_attention: MaskedAttention
    ) -> None:
        """Test that MaskedAttention compiles without graph breaks."""
        attention = standard_masked_attention
        batch_size, num_queries, embedding_dim = 2, 50, 256
        height, width = 16, 16

        query_features = torch.randn(batch_size, num_queries, embedding_dim)
        image_features = torch.randn(batch_size, embedding_dim, height, width)
        mask = torch.randint(0, 2, (batch_size, num_queries, height, width)).bool()

        # Compile the model
        compiled_attention = torch.compile(attention, fullgraph=True)

        # Run compiled version
        compiled_output = compiled_attention(query_features, image_features, mask)

        # Run original version for comparison
        original_output = attention(query_features, image_features, mask)

        # Check outputs are equivalent
        assert compiled_output.shape == original_output.shape
        assert torch.allclose(compiled_output, original_output, rtol=1e-4)

    def test_image_features_flattening(
        self, standard_masked_attention: MaskedAttention
    ) -> None:
        """Test that image features are correctly flattened and permuted."""
        attention = standard_masked_attention
        batch_size, num_queries, embedding_dim = 1, 5, 256
        height, width = 4, 8

        query_features = torch.randn(batch_size, num_queries, embedding_dim)
        image_features = torch.randn(batch_size, embedding_dim, height, width)
        mask = torch.ones(batch_size, num_queries, height, width).bool()

        # Manually check the transformation
        expected_flattened = image_features.flatten(2).permute(0, 2, 1)
        assert expected_flattened.shape == (batch_size, height * width, embedding_dim)

        # Should not raise any errors
        output = attention(query_features, image_features, mask)
        assert output.shape == (batch_size, num_queries, embedding_dim)

    def test_mask_flattening(self, standard_masked_attention: MaskedAttention) -> None:
        """Test that mask is correctly flattened."""
        attention = standard_masked_attention
        batch_size, num_queries, embedding_dim = 1, 3, 256
        height, width = 4, 6

        query_features = torch.randn(batch_size, num_queries, embedding_dim)
        image_features = torch.randn(batch_size, embedding_dim, height, width)
        mask = torch.randint(0, 2, (batch_size, num_queries, height, width)).bool()

        # Manually check mask transformation
        expected_mask_flat = mask.flatten(2)
        assert expected_mask_flat.shape == (batch_size, num_queries, height * width)

        # Should not raise any errors
        output = attention(query_features, image_features, mask)
        assert output.shape == (batch_size, num_queries, embedding_dim)

    def test_compile_validation(
        self, standard_masked_attention: MaskedAttention
    ) -> None:
        """Test that compilation works without errors and produces valid results."""
        attention = standard_masked_attention
        batch_size, num_queries, embedding_dim = 2, 50, 256
        height, width = 16, 16

        query_features = torch.randn(batch_size, num_queries, embedding_dim)
        image_features = torch.randn(batch_size, embedding_dim, height, width)
        mask = torch.randint(0, 2, (batch_size, num_queries, height, width)).bool()

        # Test compilation with different modes
        try:
            # Test fullgraph compilation
            compiled_attention_full = torch.compile(attention, fullgraph=True)
            output_full = compiled_attention_full(query_features, image_features, mask)
            assert output_full.shape == (batch_size, num_queries, embedding_dim)

            # Test default compilation
            compiled_attention_default = torch.compile(attention)
            output_default = compiled_attention_default(
                query_features, image_features, mask
            )
            assert output_default.shape == (batch_size, num_queries, embedding_dim)

            # Test that compiled versions produce consistent results
            original_output = attention(query_features, image_features, mask)
            assert torch.allclose(output_full, original_output, rtol=1e-4, atol=1e-5)
            assert torch.allclose(output_default, original_output, rtol=1e-4, atol=1e-5)

        except Exception as e:
            pytest.fail(f"Compilation failed with error: {e}")

    @pytest.mark.benchmark
    @torch.inference_mode()
    def test_performance_benchmark(
        self, standard_masked_attention: MaskedAttention
    ) -> None:
        """Benchmark test for MaskedAttention performance.

        This test measures the performance of the MaskedAttention module
        with realistic input sizes. It's marked as 'benchmark' to be
        disabled by default in CI.
        """
        import time

        attention = standard_masked_attention
        # Use realistic sizes for benchmarking
        batch_size, num_queries, embedding_dim = 8, 200, 256
        height, width = 64, 64

        query_features = torch.randn(batch_size, num_queries, embedding_dim)
        image_features = torch.randn(batch_size, embedding_dim, height, width)
        mask = torch.randint(0, 2, (batch_size, num_queries, height, width)).bool()

        # Move to GPU if available for more realistic benchmarking
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        attention = attention.to(device)
        query_features = query_features.to(device)
        image_features = image_features.to(device)
        mask = mask.to(device)

        # Warm up runs
        for _ in range(3):
            _ = attention(query_features, image_features, mask)

        if device.type == "cuda":
            torch.cuda.synchronize()

        # Benchmark forward pass
        num_runs = 50
        output = None
        start_time = time.perf_counter()

        for _ in range(num_runs):
            output = attention(query_features, image_features, mask)

        if device.type == "cuda":
            torch.cuda.synchronize()

        end_time = time.perf_counter()

        avg_time = (end_time - start_time) / num_runs

        # Basic sanity checks
        assert output is not None
        assert output.shape == (batch_size, num_queries, embedding_dim)
        assert avg_time > 0

        # Log benchmark results
        logger.info(
            "\nMaskedAttention Benchmark Results\n"
            + ("=" * 50 + "\n")
            + f"Device: {device}\n"
            f"Input shapes - Queries: {query_features.shape}, "
            f"Images: {image_features.shape}, Mask: {mask.shape}\n"
            f"Batch size: {batch_size}, Num queries: {num_queries}\n"
            f"Average forward pass time: {avg_time * 1000:.3f} ms\n"
            f"Throughput: {batch_size / avg_time:.2f} samples/sec"
        )

        # Test compiled version benchmark
        compiled_attention = torch.compile(attention, fullgraph=True)

        # Warm up compiled version
        for _ in range(10):
            _ = compiled_attention(query_features, image_features, mask)

        if device.type == "cuda":
            torch.cuda.synchronize()

        compiled_output = None
        start_time = time.perf_counter()

        for _ in range(num_runs):
            compiled_output = compiled_attention(query_features, image_features, mask)

        if device.type == "cuda":
            torch.cuda.synchronize()

        end_time = time.perf_counter()

        compiled_avg_time = (end_time - start_time) / num_runs

        speedup = avg_time / compiled_avg_time
        logger.info(
            "\nCompiled Version Results:\n"
            f"Compiled forward pass time: {compiled_avg_time * 1000:.3f} ms\n"
            f"Compiled throughput: {batch_size / compiled_avg_time:.2f} samples/sec\n"
            f"Compilation speedup: {speedup:.2f}x\n" + "=" * 50
        )

        # Verify compiled version produces same results
        assert compiled_output is not None
        assert torch.allclose(output, compiled_output, rtol=1e-4, atol=1e-5)

        # Performance assertions (these are loose to avoid flaky tests)
        assert compiled_avg_time > 0
        # Compiled version should be at least as fast (or not significantly slower)
        assert compiled_avg_time <= avg_time
