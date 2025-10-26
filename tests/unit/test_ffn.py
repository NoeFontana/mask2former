import pytest
import torch

from mask2former.modeling.common.ffn import FFN, MLP


@pytest.fixture(scope="class")
def standard_ffn() -> FFN:
    """Fixture providing a standard FFN for testing."""
    return FFN(embedding_dim=256, hidden_dim=1024)


@pytest.fixture(scope="class")
def small_ffn() -> FFN:
    """Fixture providing a small FFN for quick tests."""
    return FFN(embedding_dim=64, hidden_dim=256)


@pytest.fixture(scope="class")
def standard_mlp() -> MLP:
    """Fixture providing a standard MLP for testing."""
    return MLP(embedding_dim=256, hidden_dim=512)


class TestFFN:
    """Tests for the FFN (Feed-Forward Network) class."""

    def test_initialization(self, standard_ffn: FFN) -> None:
        """Test proper initialization of FFN."""
        ffn = standard_ffn

        # Check layers are initialized correctly
        assert isinstance(ffn.linear_1, torch.nn.Linear)
        assert isinstance(ffn.activation, torch.nn.GELU)
        assert isinstance(ffn.linear_2, torch.nn.Linear)

        # Check dimensions
        assert ffn.linear_1.in_features == 256
        assert ffn.linear_1.out_features == 1024
        assert ffn.linear_2.in_features == 1024
        assert ffn.linear_2.out_features == 256

    def test_forward_pass_shape(self, standard_ffn: FFN) -> None:
        """Test forward pass produces correct output shape."""
        ffn = standard_ffn
        batch_size, seq_len, embedding_dim = 2, 100, 256

        # Create input tensor
        input_tensor = torch.randn(batch_size, seq_len, embedding_dim)

        # Forward pass
        output = ffn(input_tensor)

        # Check output shape matches input shape
        assert output.shape == (batch_size, seq_len, embedding_dim)
        assert output.dtype == torch.float32

    def test_nonlinearity(self, standard_ffn: FFN) -> None:
        """Test that FFN applies nonlinearity (output != linear transformation)."""
        ffn = standard_ffn
        batch_size, seq_len, embedding_dim = 1, 10, 256

        input_tensor = torch.randn(batch_size, seq_len, embedding_dim)
        output = ffn(input_tensor)

        # Output should have same shape but different values due to GELU activation
        assert output.shape == input_tensor.shape
        assert not torch.allclose(output, input_tensor, rtol=1e-3)

    def test_gradient_flow(self, standard_ffn: FFN) -> None:
        """Test that gradients flow properly through the FFN module."""
        ffn = standard_ffn
        batch_size, seq_len, embedding_dim = 1, 5, 256

        input_tensor = torch.randn(
            batch_size, seq_len, embedding_dim, requires_grad=True
        )

        output = ffn(input_tensor)
        loss = output.sum()
        loss.backward()

        # Check that gradients are computed for input and parameters
        assert input_tensor.grad is not None
        assert ffn.linear_1.weight.grad is not None
        assert ffn.linear_2.weight.grad is not None

    @pytest.mark.parametrize(
        "embedding_dim,hidden_dim", [(64, 256), (128, 512), (256, 1024), (512, 2048)]
    )
    def test_different_dimensions(self, embedding_dim: int, hidden_dim: int) -> None:
        """Test with different embedding and hidden dimensions."""
        ffn = FFN(embedding_dim=embedding_dim, hidden_dim=hidden_dim)
        batch_size, seq_len = 2, 20

        input_tensor = torch.randn(batch_size, seq_len, embedding_dim)
        output = ffn(input_tensor)

        assert output.shape == (batch_size, seq_len, embedding_dim)
        assert output.dtype == torch.float32

    @pytest.mark.parametrize("batch_size,seq_len", [(1, 50), (4, 25), (8, 100)])
    def test_different_batch_and_sequence_sizes(
        self, small_ffn: FFN, batch_size: int, seq_len: int
    ) -> None:
        """Test with different batch sizes and sequence lengths."""
        ffn = small_ffn
        embedding_dim = 64

        input_tensor = torch.randn(batch_size, seq_len, embedding_dim)
        output = ffn(input_tensor)

        assert output.shape == (batch_size, seq_len, embedding_dim)
        assert output.dtype == torch.float32

    def test_deterministic(self, standard_ffn: FFN) -> None:
        """Test that FFN produces deterministic results with same inputs."""
        ffn = standard_ffn
        batch_size, seq_len, embedding_dim = 1, 5, 256

        # Set seed for reproducibility
        torch.manual_seed(42)
        input_tensor = torch.randn(batch_size, seq_len, embedding_dim)

        # Run twice with same inputs
        torch.manual_seed(42)
        output1 = ffn(input_tensor)

        torch.manual_seed(42)
        output2 = ffn(input_tensor)

        # Outputs should be identical
        assert torch.allclose(output1, output2, rtol=1e-6, atol=1e-8)

    def test_compilation(self, standard_ffn: FFN) -> None:
        """Test that FFN compiles without errors."""
        ffn = standard_ffn
        batch_size, seq_len, embedding_dim = 2, 50, 256

        input_tensor = torch.randn(batch_size, seq_len, embedding_dim)

        # Compile the model
        compiled_ffn = torch.compile(ffn, fullgraph=True)

        # Run compiled version
        compiled_output = compiled_ffn(input_tensor)

        # Run original version for comparison
        original_output = ffn(input_tensor)

        # Check outputs are equivalent
        assert compiled_output.shape == original_output.shape
        assert torch.allclose(compiled_output, original_output, rtol=1e-3, atol=1e-4)


class TestMLP:
    """Basic tests for the MLP (Multi-Layer Perceptron) class."""

    def test_initialization(self, standard_mlp: MLP) -> None:
        """Test proper initialization of MLP."""
        mlp = standard_mlp

        # Check layers are initialized correctly
        assert isinstance(mlp.linear_1, torch.nn.Linear)
        assert isinstance(mlp.activation, torch.nn.ReLU)
        assert isinstance(mlp.linear_2, torch.nn.Linear)
        assert isinstance(mlp.linear_3, torch.nn.Linear)

        # Check dimensions
        assert mlp.linear_1.in_features == 256
        assert mlp.linear_1.out_features == 512
        assert mlp.linear_2.in_features == 512
        assert mlp.linear_2.out_features == 512
        assert mlp.linear_3.in_features == 512
        assert mlp.linear_3.out_features == 256

    def test_forward_pass_shape(self, standard_mlp: MLP) -> None:
        """Test forward pass produces correct output shape."""
        mlp = standard_mlp
        batch_size, num_queries, embedding_dim = 2, 100, 256

        # Create input tensor
        input_tensor = torch.randn(batch_size, num_queries, embedding_dim)

        # Forward pass
        output = mlp(input_tensor)

        # Check output shape matches input shape
        assert output.shape == (batch_size, num_queries, embedding_dim)
        assert output.dtype == torch.float32

    def test_gradient_flow(self, standard_mlp: MLP) -> None:
        """Test that gradients flow properly through the MLP module."""
        mlp = standard_mlp
        batch_size, num_queries, embedding_dim = 1, 10, 256

        input_tensor = torch.randn(
            batch_size, num_queries, embedding_dim, requires_grad=True
        )

        output = mlp(input_tensor)
        loss = output.sum()
        loss.backward()

        # Check that gradients are computed for input and parameters
        assert input_tensor.grad is not None
        assert mlp.linear_1.weight.grad is not None
        assert mlp.linear_2.weight.grad is not None
        assert mlp.linear_3.weight.grad is not None

    @pytest.mark.parametrize(
        "embedding_dim,hidden_dim", [(64, 128), (128, 256), (256, 512)]
    )
    def test_different_dimensions(self, embedding_dim: int, hidden_dim: int) -> None:
        """Test with different embedding and hidden dimensions."""
        mlp = MLP(embedding_dim=embedding_dim, hidden_dim=hidden_dim)
        batch_size, num_queries = 2, 50

        input_tensor = torch.randn(batch_size, num_queries, embedding_dim)
        output = mlp(input_tensor)

        assert output.shape == (batch_size, num_queries, embedding_dim)
        assert output.dtype == torch.float32
