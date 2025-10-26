import torch
from torch import nn


class SelfAttention(nn.Module):
    """Self attention module.

    This module implements the Self-Attention of the transformer decoder module.

    Args:
        embedding_dim (int): Dimension of the embedding space for attention computation.
        num_head (int): Number of attention heads.
    """

    def __init__(self, embedding_dim: int, num_head: int) -> None:
        super().__init__()

        if embedding_dim % num_head:
            raise ValueError(
                f"embedding_dim must be divisible by num_head, but got "
                f"embedding_dim={embedding_dim} and num_head={num_head}"
            )

        self.num_head = num_head
        self.head_embedding_dim = embedding_dim // num_head

        self.out_proj = (
            nn.Linear(embedding_dim, embedding_dim) if num_head > 1 else nn.Identity()
        )

        self.query_projector = nn.Linear(
            in_features=embedding_dim, out_features=embedding_dim
        )
        self.key_projector = nn.Linear(
            in_features=embedding_dim, out_features=embedding_dim
        )
        self.value_projector = nn.Linear(
            in_features=embedding_dim, out_features=embedding_dim
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize learnable parameters."""
        nn.init.xavier_uniform_(self.query_projector.weight)
        nn.init.xavier_uniform_(self.key_projector.weight)
        nn.init.xavier_uniform_(self.value_projector.weight)
        if isinstance(self.out_proj, nn.Linear):
            nn.init.xavier_uniform_(self.out_proj.weight)

        nn.init.constant_(self.query_projector.bias, 0.0)
        nn.init.constant_(self.key_projector.bias, 0.0)
        nn.init.constant_(self.value_projector.bias, 0.0)
        if isinstance(self.out_proj, nn.Linear):
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self, query_features: torch.Tensor, pos_query_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass of the self attention module.

        Args:
            query_features (torch.Tensor): Query features tensor.
                Shape: (batch_size, num_queries, embedding_dim)
            pos_query_embeddings (torch.Tensor): Positional embeddings for
                query features.
                Shape: (batch_size, num_queries, embedding_dim)

        Returns:
            torch.Tensor: Attended feature tensor.
                Shape: (batch_size, num_queries, embedding_dim)
        """
        batch_size, num_queries, embedding_dim = query_features.shape

        q = k = query_features + pos_query_embeddings
        q = self.query_projector(q)
        k = self.key_projector(k)
        v = self.value_projector(query_features)

        # Reshape for multi-head attention
        q = q.view(
            batch_size, num_queries, self.num_head, self.head_embedding_dim
        ).transpose(1, 2)
        k = k.view(
            batch_size, num_queries, self.num_head, self.head_embedding_dim
        ).transpose(1, 2)
        v = v.view(
            batch_size, num_queries, self.num_head, self.head_embedding_dim
        ).transpose(1, 2)

        attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v)

        attn_out = (
            attn_out.transpose(1, 2)
            .contiguous()
            .view(batch_size, num_queries, embedding_dim)
        )

        return self.out_proj(attn_out)
