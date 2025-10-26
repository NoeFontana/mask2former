import torch
from torch import nn


class FFN(nn.Module):
    """Feed-Forward Network module.

    This module implements the Feed-Forward Network of the transformer decoder module.

    Args:
        embedding_dim (int): Dimension of the embedding space for attention computation.
        hidden_dim (int): Dimension of the hidden layer.
    """

    def __init__(self, embedding_dim: int, hidden_dim: int) -> None:
        super().__init__()

        self.linear_1 = nn.Linear(in_features=embedding_dim, out_features=hidden_dim)
        self.activation = nn.GELU()
        self.linear_2 = nn.Linear(in_features=hidden_dim, out_features=embedding_dim)

    def forward(self, queries: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.activation(self.linear_1(queries)))


class MLP(nn.Module):
    """MLP layer that predicts mask embeddings from query features."""

    def __init__(self, embedding_dim: int, hidden_dim: int) -> None:
        super().__init__()

        self.linear_1 = nn.Linear(in_features=embedding_dim, out_features=hidden_dim)
        self.activation = nn.ReLU()
        self.linear_2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.linear_3 = nn.Linear(in_features=hidden_dim, out_features=embedding_dim)

    def forward(self, query_features: torch.Tensor) -> torch.Tensor:
        """query_features: (batch_size, num_queries, embedding_dim)

        Returns:
            torch.Tensor: Mask embeddings tensor.
                Shape: (batch_size, num_queries, embedding_dim)
        """
        return self.linear_3(
            self.activation(
                self.linear_2(self.activation(self.linear_1(query_features)))
            )
        )
