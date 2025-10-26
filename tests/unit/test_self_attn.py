import logging

import pytest
import torch

from mask2former.modeling.attn import SelfAttention

logger = logging.getLogger(__name__)


@pytest.fixture(scope="class")
def standard_self_attention() -> SelfAttention:
    """Fixture providing a standard SelfAttention for testing."""
    return SelfAttention(embedding_dim=256, num_head=8)


@pytest.fixture(scope="class")
def small_self_attention() -> SelfAttention:
    """Fixture providing a small SelfAttention for quick tests."""
    return SelfAttention(embedding_dim=64, num_head=2)


class TestSelfAttention:
    """Tests for the SelfAttention class."""

    def test_initialization(self, standard_self_attention: SelfAttention) -> None:
        """Test proper initialization of SelfAttention."""
        attention = standard_self_attention

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

    def test_forward_pass_shape(self, standard_self_attention: SelfAttention) -> None:
        """Test forward pass produces correct output shape."""
        attention = standard_self_attention
        batch_size, num_queries, embedding_dim = 2, 100, 256

        # Create input tensor
        query_features = torch.randn(batch_size, num_queries, embedding_dim)
        pos_query_embeddings = torch.randn(batch_size, num_queries, embedding_dim)

        # Forward pass
        output = attention(query_features, pos_query_embeddings)

        # Check output shape matches input shape
        assert output.shape == (batch_size, num_queries, embedding_dim)
        assert output.dtype == torch.float32

    def test_residual_connection(self, standard_self_attention: SelfAttention) -> None:
        """Test that residual connection is applied correctly."""
        attention = standard_self_attention
        batch_size, num_queries, embedding_dim = 1, 10, 256

        query_features = torch.randn(batch_size, num_queries, embedding_dim)
        pos_query_embeddings = torch.randn(batch_size, num_queries, embedding_dim)
        output = attention(query_features, pos_query_embeddings)

        # The output should not be equal to input due to attention computation
        # but should have the same shape due to residual connection
        assert output.shape == query_features.shape
        assert not torch.allclose(output, query_features, rtol=1e-3)

    def test_self_attention_properties(
        self, standard_self_attention: SelfAttention
    ) -> None:
        """Test that self-attention has expected properties."""
        attention = standard_self_attention
        batch_size, num_queries, embedding_dim = 1, 5, 256

        # Create deterministic input
        torch.manual_seed(42)
        query_features = torch.randn(batch_size, num_queries, embedding_dim)
        pos_query_embeddings = torch.randn(batch_size, num_queries, embedding_dim)

        output = attention(query_features, pos_query_embeddings)

        # Output should have same shape as input
        assert output.shape == query_features.shape

        # Self-attention should be permutation equivariant:
        # If we permute the input sequence, the output should be permuted the same way
        perm_indices = torch.tensor([4, 0, 2, 1, 3])
        permuted_input = query_features[:, perm_indices, :]
        permuted_output = attention(
            permuted_input, pos_query_embeddings[:, perm_indices, :]
        )

        # Check that attention(permuted_input) == permuted(attention(input))
        expected_permuted_output = output[:, perm_indices, :]
        assert torch.allclose(
            permuted_output, expected_permuted_output, rtol=1e-4, atol=1e-5
        )

    def test_gradient_flow(self, standard_self_attention: SelfAttention) -> None:
        """Test that gradients flow properly through the attention module."""
        attention = standard_self_attention
        batch_size, num_queries, embedding_dim = 1, 10, 256

        query_features = torch.randn(
            batch_size, num_queries, embedding_dim, requires_grad=True
        )
        pos_query_embeddings = torch.randn(batch_size, num_queries, embedding_dim)

        output = attention(query_features, pos_query_embeddings)
        loss = output.sum()
        loss.backward()

        # Check that gradients are computed for input
        assert query_features.grad is not None

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
        attention = SelfAttention(embedding_dim=embedding_dim, num_head=num_head)
        batch_size, num_queries = 2, 20

        query_features = torch.randn(batch_size, num_queries, embedding_dim)
        pos_query_embeddings = torch.randn(batch_size, num_queries, embedding_dim)
        output = attention(query_features, pos_query_embeddings)

        assert output.shape == (batch_size, num_queries, embedding_dim)
        assert output.dtype == torch.float32

    @pytest.mark.parametrize("batch_size,num_queries", [(1, 50), (3, 100), (8, 200)])
    def test_different_batch_and_query_sizes(
        self,
        small_self_attention: SelfAttention,
        batch_size: int,
        num_queries: int,
    ) -> None:
        """Test with different batch sizes and number of queries."""
        attention = small_self_attention
        embedding_dim = 64

        query_features = torch.randn(batch_size, num_queries, embedding_dim)
        pos_query_embeddings = torch.randn(batch_size, num_queries, embedding_dim)
        output = attention(query_features, pos_query_embeddings)

        assert output.shape == (batch_size, num_queries, embedding_dim)
        assert output.dtype == torch.float32

    def test_attention_deterministic(
        self, standard_self_attention: SelfAttention
    ) -> None:
        """Test that attention produces deterministic results with same inputs."""
        attention = standard_self_attention
        batch_size, num_queries, embedding_dim = 1, 5, 256

        # Set seeds for reproducibility
        torch.manual_seed(42)
        query_features = torch.randn(batch_size, num_queries, embedding_dim)
        pos_query_embeddings = torch.randn(batch_size, num_queries, embedding_dim)

        # Run twice with same inputs
        torch.manual_seed(42)
        output1 = attention(query_features, pos_query_embeddings)

        torch.manual_seed(42)
        output2 = attention(query_features, pos_query_embeddings)

        # Outputs should be identical
        assert torch.allclose(output1, output2, rtol=1e-6, atol=1e-8)

    def test_self_attention_compilation(
        self, standard_self_attention: SelfAttention
    ) -> None:
        """Test that SelfAttention compiles without graph breaks."""
        attention = standard_self_attention
        batch_size, num_queries, embedding_dim = 2, 50, 256

        query_features = torch.randn(batch_size, num_queries, embedding_dim)
        pos_query_embeddings = torch.randn(batch_size, num_queries, embedding_dim)

        # Compile the model
        compiled_attention = torch.compile(attention, fullgraph=True)

        # Run compiled version
        compiled_output = compiled_attention(query_features, pos_query_embeddings)

        # Run original version for comparison
        original_output = attention(query_features, pos_query_embeddings)

        # Check outputs are equivalent
        assert compiled_output.shape == original_output.shape
        assert torch.allclose(compiled_output, original_output, rtol=1e-4)

    def test_multi_head_validation(self) -> None:
        """Test multi-head attention parameter validation."""
        # Test valid configurations
        attention_valid = SelfAttention(embedding_dim=256, num_head=8)
        assert attention_valid.num_head == 8
        assert attention_valid.head_embedding_dim == 32

        # Test invalid configuration (not divisible)
        with pytest.raises(
            ValueError, match="embedding_dim must be divisible by num_head"
        ):
            SelfAttention(embedding_dim=257, num_head=8)

    def test_single_head_attention(self) -> None:
        """Test single-head attention (should use Identity for out_proj)."""
        attention = SelfAttention(embedding_dim=64, num_head=1)

        assert attention.num_head == 1
        assert attention.head_embedding_dim == 64
        assert isinstance(attention.out_proj, torch.nn.Identity)

        batch_size, num_queries = 2, 10

        query_features = torch.randn(batch_size, num_queries, 64)
        pos_query_embeddings = torch.randn(batch_size, num_queries, 64)

        output = attention(
            query_features,
            pos_query_embeddings,
        )
        assert output.shape == (batch_size, num_queries, 64)

    @pytest.mark.parametrize("num_head", [1, 2, 4, 8, 16])
    def test_different_head_counts(self, num_head: int) -> None:
        """Test with different numbers of attention heads."""
        embedding_dim = 128
        attention = SelfAttention(embedding_dim=embedding_dim, num_head=num_head)

        batch_size, num_queries = 2, 20
        query_features = torch.randn(batch_size, num_queries, embedding_dim)
        pos_query_embeddings = torch.randn(batch_size, num_queries, embedding_dim)

        output = attention(query_features, pos_query_embeddings)

        assert output.shape == (batch_size, num_queries, embedding_dim)
        assert attention.head_embedding_dim == embedding_dim // num_head

    def test_head_reshaping(self) -> None:
        """Test that tensors are correctly reshaped for multi-head attention."""
        attention = SelfAttention(embedding_dim=256, num_head=8)

        batch_size, num_queries = 1, 5
        query_features = torch.randn(batch_size, num_queries, 256)
        pos_query_embeddings = torch.randn(batch_size, num_queries, 256)

        # Test that forward pass completes successfully with correct shapes
        output = attention(query_features, pos_query_embeddings)
        assert output.shape == (batch_size, num_queries, 256)

    def test_compile_validation(self, standard_self_attention: SelfAttention) -> None:
        """Test that compilation works without errors and produces valid results."""
        attention = standard_self_attention
        batch_size, num_queries, embedding_dim = 2, 50, 256

        query_features = torch.randn(batch_size, num_queries, embedding_dim)
        pos_query_embeddings = torch.randn(batch_size, num_queries, embedding_dim)

        # Test compilation with different modes
        try:
            # Test fullgraph compilation
            compiled_attention_full = torch.compile(attention, fullgraph=True)
            output_full = compiled_attention_full(query_features, pos_query_embeddings)
            assert output_full.shape == (batch_size, num_queries, embedding_dim)

            # Test default compilation
            compiled_attention_default = torch.compile(attention)
            output_default = compiled_attention_default(
                query_features, pos_query_embeddings
            )
            assert output_default.shape == (batch_size, num_queries, embedding_dim)

            # Test that compiled versions produce consistent results
            original_output = attention(query_features, pos_query_embeddings)
            assert torch.allclose(output_full, original_output, rtol=1e-4, atol=1e-5)
            assert torch.allclose(output_default, original_output, rtol=1e-4, atol=1e-5)

        except Exception as e:
            pytest.fail(f"Compilation failed with error: {e}")

    @pytest.mark.benchmark
    @pytest.mark.parametrize("num_queries", [100, 200, 300])
    @torch.inference_mode()
    def test_performance_benchmark(
        self, standard_self_attention: SelfAttention, num_queries: int
    ) -> None:
        """Benchmark test for SelfAttention performance.

        This test measures the performance of the SelfAttention module
        with different sequence lengths. It's marked as 'benchmark' to be
        disabled by default in CI.
        """
        import time

        attention = standard_self_attention
        batch_size, embedding_dim = 8, 256

        query_features = torch.randn(batch_size, num_queries, embedding_dim)
        pos_query_embeddings = torch.randn(batch_size, num_queries, embedding_dim)

        # Move to GPU if available for more realistic benchmarking
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        attention = attention.to(device)
        query_features = query_features.to(device)

        # Warm up runs
        for _ in range(10):
            _ = attention(query_features, pos_query_embeddings)

        if device.type == "cuda":
            torch.cuda.synchronize()

        # Benchmark forward pass
        num_runs = 100
        output = None
        start_time = time.perf_counter()

        for _ in range(num_runs):
            output = attention(query_features, pos_query_embeddings)

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
            "\nSelfAttention Benchmark Results\n"
            + ("=" * 50 + "\n")
            + f"Device: {device}\n"
            f"Input shape: {query_features.shape}\n"
            f"Batch size: {batch_size}, Sequence length: {num_queries}\n"
            f"Average forward pass time: {avg_time * 1000:.3f} ms\n"
            f"Throughput: {batch_size / avg_time:.2f} samples/sec"
        )

        # Test compiled version benchmark
        compiled_attention = torch.compile(attention, fullgraph=True)

        # Warm up compiled version
        for _ in range(10):
            _ = compiled_attention(query_features, pos_query_embeddings)

        if device.type == "cuda":
            torch.cuda.synchronize()

        compiled_output = None
        start_time = time.perf_counter()

        for _ in range(num_runs):
            compiled_output = compiled_attention(query_features, pos_query_embeddings)

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

        # Compiled version produces same results
        assert compiled_output is not None
        assert torch.allclose(output, compiled_output, rtol=1e-4, atol=1e-5)

        assert compiled_avg_time > 0
        if compiled_avg_time > avg_time:
            logger.warning(
                "Compiled version was not faster than uncompiled version.\n"
                f"Uncompiled: {avg_time * 1000:.3f} ms\n"
                f"Compiled: {compiled_avg_time * 1000:.3f} ms"
            )
