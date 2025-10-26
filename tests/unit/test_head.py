"""Unit tests for Mask2Former heads."""

import pytest
import torch

from mask2former.modeling.common.head import ClassificationHead


@pytest.fixture(scope="class")
def standard_classification_head() -> ClassificationHead:
    """Fixture providing a standard ClassificationHead for testing."""
    return ClassificationHead(hidden_dim=256, num_classes=80)


@pytest.fixture(scope="class")
def small_classification_head() -> ClassificationHead:
    """Fixture providing a small ClassificationHead for quick tests."""
    return ClassificationHead(hidden_dim=128, num_classes=20)


class TestClassificationHead:
    """Tests for the ClassificationHead class."""

    def test_initialization(
        self, standard_classification_head: ClassificationHead
    ) -> None:
        """Test proper initialization of ClassificationHead."""
        head = standard_classification_head

        assert head.hidden_dim == 256
        assert head.num_classes == 80
        assert isinstance(head.classifier, torch.nn.Linear)
        assert head.classifier.in_features == 256
        assert head.classifier.out_features == 81  # num_classes + 1 (no-object)

    def test_forward_pass(
        self,
        standard_classification_head: ClassificationHead,
    ) -> None:
        """Test forward pass through classification head."""
        head = standard_classification_head
        batch_size, num_queries, hidden_dim = 2, 100, 256
        query_embeddings = torch.randn(batch_size, num_queries, hidden_dim)

        # Forward pass
        class_logits = head(query_embeddings)

        # Check output shape
        assert class_logits.shape == (batch_size, num_queries, 81)
        assert class_logits.dtype == torch.float32

    def test_gradient_flow(
        self,
        standard_classification_head: ClassificationHead,
    ) -> None:
        """Test that gradients flow properly through the head."""
        head = standard_classification_head
        query_embeddings = torch.randn(1, 100, 256, requires_grad=True)

        class_logits = head(query_embeddings)
        loss = class_logits.sum()
        loss.backward()

        # Check that gradients are computed
        assert query_embeddings.grad is not None
        assert head.classifier.weight.grad is not None

    @pytest.mark.parametrize("object_prior_prob", [0.01, 0.1, 0.3])
    @pytest.mark.parametrize("num_classes", [1, 10])
    def test_reset_parameters_bias_probabilities(
        self, object_prior_prob: float, num_classes: int
    ) -> None:
        """Test that reset_parameters sets bias to achieve expected probabilities."""
        import math

        head = ClassificationHead(
            hidden_dim=256, num_classes=num_classes, object_prior_prob=object_prior_prob
        )

        bias = head.classifier.bias.detach()
        obj_biases = bias[:-1]  # Object class biases
        no_obj_bias = bias[-1]  # No-object class bias

        expected_no_obj_bias = math.log(
            num_classes * (1 - object_prior_prob) / object_prior_prob
        )

        # Check that bias values match expectations
        torch.testing.assert_close(obj_biases, torch.zeros_like(obj_biases))
        torch.testing.assert_close(no_obj_bias, torch.tensor(expected_no_obj_bias))

        # Test softmax probability with zero input (only bias contributes)
        zero_input = torch.zeros(1, 1, 256)  # (batch, queries, hidden_dim)
        logits = head(zero_input)  # Shape: (1, 1, num_classes + 1)

        # Apply softmax to get probabilities
        probabilities = torch.softmax(logits, dim=-1)  # Shape: (1, 1, num_classes + 1)

        # Extract probabilities
        obj_probs = probabilities[0, 0, :-1]  # Object class probabilities
        no_obj_prob = probabilities[0, 0, -1]  # No-object probability

        expected_obj_prob_per_class = object_prior_prob / num_classes
        expected_no_obj_prob_actual = 1 - object_prior_prob

        # Check that each object class has equal probability
        for obj_prob in obj_probs:
            torch.testing.assert_close(
                obj_prob,
                torch.tensor(expected_obj_prob_per_class),
                rtol=1e-3,
                atol=1e-4,
            )

        # Check no-object probability
        torch.testing.assert_close(
            no_obj_prob,
            torch.tensor(expected_no_obj_prob_actual),
            rtol=1e-3,
            atol=1e-4,
        )

        # Verify probabilities sum to 1
        total_prob = probabilities.sum()
        torch.testing.assert_close(total_prob, torch.tensor(1.0), rtol=1e-6, atol=1e-6)

        # Verify total object probability matches prior
        total_obj_prob = obj_probs.sum()
        torch.testing.assert_close(
            total_obj_prob, torch.tensor(object_prior_prob), rtol=1e-3, atol=1e-4
        )

    def test_classification_head_compilation(
        self,
        standard_classification_head: ClassificationHead,
    ) -> None:
        """Test that ClassificationHead compiles without graph breaks."""
        head = standard_classification_head
        query_embeddings = torch.randn(2, 100, 256)

        # Compile the model
        compiled_head = torch.compile(head, fullgraph=True)

        # Run compiled version
        compiled_output = compiled_head(query_embeddings)

        # Run original version for comparison
        original_output = head(query_embeddings)

        # Check outputs are equivalent
        assert compiled_output.shape == original_output.shape
        assert torch.allclose(compiled_output, original_output, rtol=1e-4)

    @pytest.mark.parametrize("batch_size,num_queries", [(1, 50), (2, 100), (4, 200)])
    def test_different_batch_sizes(
        self,
        small_classification_head: ClassificationHead,
        batch_size: int,
        num_queries: int,
    ) -> None:
        """Test with different input sizes."""
        head = small_classification_head
        query_embeddings = torch.randn(batch_size, num_queries, head.hidden_dim)

        class_logits = head(query_embeddings)

        # Check output shape matches input batch and query dimensions
        expected_classes = head.num_classes + 1  # 20 + 1 = 21
        assert class_logits.shape == (batch_size, num_queries, expected_classes)
        assert class_logits.dtype == torch.float32
