import torch
from torch import nn


class MaskedAttention(nn.Module):
    """Masked attention module.

    This module implements a masked attention mechanism where queries attend to
    image features through key-value projections, with attention masked by the
    provided mask tensor.

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

    def forward(
        self,
        query_features: torch.Tensor,
        image_features: torch.Tensor,
        mask: torch.Tensor,
        pos_query_embeddings: torch.Tensor,
        pos_image_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the masked attention module.

        Args:
            query_features (torch.Tensor): Query features tensor.
                Shape: (batch_size, num_queries, embedding_dim)
            image_features (torch.Tensor): Image feature tensor.
                Shape: (batch_size, embedding_dim, height, width)
            mask (torch.Tensor): Attention mask tensor (bool) to control which features
                the queries can attend to. False positions will be ignored.
                Shape: (batch_size, num_queries, height, width)
            pos_query_embeddings (torch.Tensor): Query feature positional embeddings.
                Shape: (batch_size, num_queries, embedding_dim)
            pos_image_embeddings (torch.Tensor): Image feature positional embeddings.
                Shape: (batch_size, (height * width), embedding_dim)

        Returns:
            torch.Tensor: Attended feature tensor.
                Shape: (batch_size, num_queries, embedding_dim)
        """
        batch_size, embedding_dim, height, width = image_features.shape
        num_queries = query_features.shape[1]
        spatial_size = height * width

        # Reshape for attention (N, C, H, W) -> (N, H*W, C)
        image_features = image_features.view(
            batch_size, embedding_dim, spatial_size
        ).transpose(-2, -1)

        q = self.query_projector(query_features + pos_query_embeddings)
        k = self.key_projector(image_features + pos_image_embeddings)
        v = self.value_projector(image_features)

        # Reshape for multi-head attention
        q = q.view(
            batch_size, num_queries, self.num_head, self.head_embedding_dim
        ).transpose(1, 2)
        k = k.view(
            batch_size, spatial_size, self.num_head, self.head_embedding_dim
        ).transpose(1, 2)
        v = v.view(
            batch_size, spatial_size, self.num_head, self.head_embedding_dim
        ).transpose(1, 2)

        attn_mask = mask.view(batch_size, 1, num_queries, spatial_size)
        attn_out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask
        )

        attn_out = (
            attn_out.transpose(1, 2)
            .contiguous()
            .view(batch_size, num_queries, embedding_dim)
        )

        return self.out_proj(attn_out)


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
