import torch
from torch import nn


class MaskedAttention(nn.Module):
    """Masked attention module.

    This module implements a masked attention mechanism where queries attend to
    image features through key-value projections, with attention masked by the
    provided mask tensor.

    Args:
        embedding_dim (int): Dimension of the embedding space for attention computation.
    """

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()

        self.query_projector = nn.Linear(
            in_features=embedding_dim, out_features=embedding_dim
        )
        self.key_projector = nn.Linear(
            in_features=embedding_dim, out_features=embedding_dim
        )
        self.value_projector = nn.Linear(
            in_features=embedding_dim, out_features=embedding_dim
        )

    def forward(
        self,
        query_features: torch.Tensor,
        image_features: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the masked attention module.

        Args:
            query_features (torch.Tensor): Query features tensor.
                Shape: (batch_size, num_queries, embedding_dim)
            image_features (torch.Tensor): Image feature tensor.
                Shape: (batch_size, embedding_dim, height, width)
            mask (torch.Tensor): Attention mask tensor (bool) to control which features
                the queries can attend to. False positions will be masked.
                Shape: (batch_size, num_queries, height, width)

        Returns:
            torch.Tensor: Attended feature tensor.
                Shape: (batch_size, num_queries, embedding_dim)
        """
        image_features = image_features.flatten(2).permute(
            0, 2, 1
        )  # (N, H*W, embedding_dim)
        mask = mask.flatten(2)  # (N, num_queries, H*W)

        q = self.query_projector(query_features)
        k = self.key_projector(image_features)
        v = self.value_projector(image_features)

        attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, mask)

        return attn_out + query_features
