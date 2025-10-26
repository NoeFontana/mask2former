"""Mask2Former classification and segmentation heads."""

import math

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """
    Classification head for Mask2Former that predicts class logits for each query.

    This head takes query embeddings and produces class predictions including a
    "no-object" class for empty queries.
    """

    def __init__(
        self,
        num_classes: int,
        hidden_dim: int = 256,
        object_prior_prob: float = 0.01,
    ) -> None:
        """Initialize the classification head.

        Args:
            num_classes: Number of semantic classes (excluding no-object class)
            hidden_dim: Hidden dimension of query embeddings
            object_prior_prob: Prior probability of having an object for bias
        """
        super().__init__()
        self.hidden_dim: int = hidden_dim
        self.num_classes: int = num_classes
        self.classifier: nn.Linear = nn.Linear(hidden_dim, num_classes + 1)

        self._reset_parameters(object_prior_prob)

    def _reset_parameters(self, object_prior_prob: float, gain: float = 1.0) -> None:
        """Initialize parameters.

        Args:
            object_prior_prob: Prior probability of a prediction being an object
            gain: Gain factor for Xavier uniform weight initialization
        """
        nn.init.xavier_uniform_(self.classifier.weight, gain=gain)
        if self.classifier.bias is not None:
            # Initialize bias to achieve desired prior probability
            # For softmax with uniform weights, we want P(object) = object_prior_prob
            # i.e. exp(b_obj) / (K*exp(b_obj) + exp(b_no_obj)) = object_prior_prob
            # Where K is the number of object classes
            # Setting b_obj = 0 and b_no_obj = log(K * (1-p)/p) achieves this
            bias_value = math.log(
                self.num_classes * (1 - object_prior_prob) / object_prior_prob
            )

            # Object classes get bias = 0, no-object class gets positive bias
            nn.init.constant_(self.classifier.bias[:-1], 0.0)
            nn.init.constant_(self.classifier.bias[-1], bias_value)

    def forward(self, query_embeddings: torch.Tensor) -> torch.Tensor:
        """Forward pass through classification head.

        Args:
            query_embeddings: Query embeddings of shape
                (batch_size, num_queries, hidden_dim)

        Returns:
            Class logits of shape (batch_size, num_queries, num_classes + 1)
        """
        # Direct classification without reshaping - works with 3D tensor
        class_logits = self.classifier(query_embeddings)

        return class_logits
