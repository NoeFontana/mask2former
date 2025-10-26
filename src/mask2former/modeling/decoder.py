import torch
from torch import nn

from mask2former.modeling.common.ffn import FFN, MLP
from mask2former.modeling.pe import sine_pe_2d


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


def generate_mask_logits(
    output_size: tuple[int, int],
    query_features: torch.Tensor,
    pixel_features: torch.Tensor,
    mask_embedder: MLP,
) -> torch.Tensor:
    """Convert query features to output size mask probabilities.

    Args:
        output_size (tuple[int, int]): Output (height, width).
        query_features (torch.Tensor): Query features tensor.
            Shape: (batch_size, num_queries, embedding_dim)
        pixel_features (torch.Tensor): Pixel features tensor.
            Shape: (batch_size, embedding_dim, height, width)
        mask_embedder (MLP): MLP module to convert query features to mask embeddings.

    Returns:
        torch.Tensor: Mask probabilities tensor.
            Shape: (batch_size, num_queries, output_size[0], output_size[1])
    """
    mask_embed = mask_embedder(query_features)
    # Mask logits are obtained via a dot product over the embedding dimension
    outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, pixel_features)

    # Interpolate to target size if different from pixel features size
    current_size = pixel_features.shape[-2:]
    if output_size != current_size:
        outputs_mask = torch.nn.functional.interpolate(
            outputs_mask,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )

    return outputs_mask


def generate_attention_mask(
    mask_logits: torch.Tensor,
    attn_mask_target_size: tuple[int, int],
) -> torch.Tensor:
    """Generate boolean attention mask for cross-attention.

    Args:
        mask_logits (torch.Tensor): Raw mask logits.
            Shape: (batch_size, num_queries, height, width)
        attn_mask_target_size (tuple[int, int]): Target size for attention mask.

    Returns:
        torch.Tensor: Boolean attention mask.
            Shape: (batch_size, num_queries, target_height * target_width)
    """
    # Interpolate to attention target size
    attn_mask = torch.nn.functional.interpolate(
        mask_logits,
        size=attn_mask_target_size,
        mode="bilinear",
        align_corners=False,
    )

    # Convert to boolean mask for attention
    # True positions are unchanged, False positions are not allowed to attend
    # (Torch 2.9 scaled_dot_product_attention uses True for keep, False for ignore)
    attn_mask = (attn_mask.sigmoid().flatten(2) >= 0.5).bool()

    return attn_mask.detach()


class DecoderLayer(nn.Module):
    """Single Transformer Decoder Layer for Mask2Former.

    The layer consists of the following sequence of operations:

        Masked Attention -> Residual Connection -> Norm ->
        Self-Attention -> Residual Connection -> Norm ->
        Feed-Forward Network -> Residual Connection -> Norm
    """

    def __init__(
        self,
        embedding_dim: int,
        num_head: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()

        self.masked_attention = MaskedAttention(embedding_dim, num_head)
        self.self_attention = SelfAttention(embedding_dim, num_head)
        self.ffn = FFN(embedding_dim, hidden_dim)

        self.norm_1 = nn.LayerNorm(embedding_dim)
        self.norm_2 = nn.LayerNorm(embedding_dim)
        self.norm_3 = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        query_features: torch.Tensor,
        image_features: torch.Tensor,
        masks: torch.Tensor,
        pos_query_embeddings: torch.Tensor,
        pos_image_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the decoder layer.
        Args:
            query_features (torch.Tensor): Query features tensor.
                Shape: (batch_size, num_queries, embedding_dim)
            image_features (torch.Tensor): Image feature tensor.
                Shape: (batch_size, embedding_dim, height, width)
            masks (torch.Tensor): Attention masks, typically at 1/32 scale.
                Shape: (batch_size, num_queries, height, width)
            pos_query_embeddings (torch.Tensor): Positional embeddings for
                query features.
                Shape: (batch_size, num_queries, embedding_dim)
            pos_image_embeddings (torch.Tensor): Positional embeddings for
                image features.
                Shape: (batch_size, (height, width), embedding_dim)
        Returns:
            torch.Tensor: Output tensor after passing through the decoder layer.
                Shape: (batch_size, num_queries, embedding_dim)
        """
        query_features = self.norm_1(
            query_features
            + self.masked_attention(
                query_features,
                image_features,
                masks,
                pos_query_embeddings,
                pos_image_embeddings,
            )
        )
        query_features = self.norm_2(
            query_features + self.self_attention(query_features, pos_query_embeddings)
        )
        query_features = self.norm_3(query_features + self.ffn(query_features))
        return query_features


class MultiScaleDecoderLayer(nn.Module):
    """Multi-Scale Transformer Decoder Layer for Mask2Former."""

    def __init__(
        self,
        embedding_dim: int,
        num_head: int,
        hidden_dim: int,
        mask_predictor: MLP,
        num_feature_levels: int = 3,
    ) -> None:
        super().__init__()

        self.decoder = nn.ModuleList(
            [
                DecoderLayer(embedding_dim, num_head, hidden_dim)
                for _ in range(num_feature_levels)
            ]
        )
        self.mask_predictor = mask_predictor
        self.num_feature_levels = num_feature_levels
        self.num_head = num_head

    def forward(
        self,
        query_features: torch.Tensor,
        image_features_list: list[torch.Tensor],
        mask_features: torch.Tensor,
        pos_query_embeddings: torch.Tensor,
        pos_image_embeddings_list: list[torch.Tensor],
        return_auxiliary_masks: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass of the multi-scale decoder layer."""
        batch_size, num_queries = query_features.shape[:2]

        # Only allocate auxiliary masks tensor if needed
        auxiliary_masks = None
        if return_auxiliary_masks:
            max_spatial_size = max(
                feat.shape[-2] * feat.shape[-1] for feat in image_features_list
            )
            auxiliary_masks = torch.empty(
                self.num_feature_levels,
                batch_size,
                num_queries,
                max_spatial_size,
                device=query_features.device,
                dtype=query_features.dtype,
            )

        for i in range(self.num_feature_levels):
            image_features = image_features_list[i]
            pos_image_embeddings = pos_image_embeddings_list[i]
            decoder_layer = self.decoder[i]

            image_height, image_width = image_features.shape[-2:]

            # Generate mask logits (not probabilities yet)
            mask_logits = generate_mask_probabilities(
                (image_height, image_width),
                query_features,
                mask_features,
                self.mask_predictor,
            )

            # Store auxiliary masks if requested
            if auxiliary_masks is not None:
                masks_flat = mask_logits.view(batch_size, num_queries, -1)
                spatial_size = masks_flat.shape[-1]
                auxiliary_masks[i, :, :, :spatial_size] = masks_flat

            # Generate boolean attention mask
            attention_mask = generate_attention_mask(
                mask_logits, (image_height, image_width)
            )

            query_features = decoder_layer(
                query_features,
                image_features,
                attention_mask,
                pos_query_embeddings,
                pos_image_embeddings,
            )

        return query_features, auxiliary_masks


class TransformerDecoder(nn.Module):
    """Transformer Decoder for Mask2Former."""

    def __init__(
        self,
        num_layers: int,
        embedding_dim: int,
        num_query: int,
        num_head: int,
        hidden_dim: int,
        num_classes: int,
        num_feature_levels: int = 3,
    ) -> None:
        super().__init__()

        self.num_layers = num_layers
        self.num_feature_levels = num_feature_levels
        self.num_head = num_head

        self.mask_embedder = MLP(embedding_dim, hidden_dim=256)

        # Add decoder normalization (matches reference)
        self.decoder_norm = nn.LayerNorm(embedding_dim)

        self.initial_queries = nn.Parameter(torch.randn(1, num_query, embedding_dim))
        self.pos_query_embed = nn.Parameter(torch.randn(1, num_query, embedding_dim))

        self.layers = nn.ModuleList(
            [
                MultiScaleDecoderLayer(
                    embedding_dim,
                    num_head,
                    hidden_dim,
                    self.mask_embedder,
                    num_feature_levels,
                )
                for _ in range(num_layers)
            ]
        )

        # Prediction heads (matches reference structure)
        self.classifier = nn.Linear(embedding_dim, num_classes + 1)

        # Input configuration
        self.input_height = 512
        self.input_width = 512
        self.output_divisors = (32, 16, 8)

        # Register positional embeddings as buffers
        for i, divisor in enumerate(self.output_divisors):
            ape = (
                sine_pe_2d(
                    embedding_dim,
                    self.input_height // divisor,
                    self.input_width // divisor,
                )
                .permute(0, 2, 3, 1)
                .view(1, -1, embedding_dim)
            )
            self.register_buffer(f"ape_{i}", ape)

    def forward_prediction_heads(
        self,
        query_features: torch.Tensor,
        mask_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward prediction heads ."""
        # Apply decoder normalization
        decoder_output = self.decoder_norm(query_features)

        # Generate class predictions
        outputs_class = self.classifier(decoder_output)

        # Generate mask predictions
        mask_height, mask_width = mask_features.shape[-2:]
        outputs_mask = generate_mask_probabilities(
            (mask_height, mask_width),
            decoder_output,
            mask_features,
            self.mask_embedder,
        )

        return outputs_class, outputs_mask

    def forward(
        self,
        image_features_list: list[torch.Tensor],
        mask_features: torch.Tensor,
        return_auxiliary_masks: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Forward pass optimized for torch.compile and torch.export."""

        query_features = self.initial_queries.expand(mask_features.shape[0], -1, -1)

        # Build positional embeddings list
        pos_embeddings_list = [
            getattr(self, f"ape_{i}") for i in range(self.num_feature_levels)
        ]

        # Pre-allocate auxiliary masks if needed
        all_aux_masks = torch.empty(0, device=query_features.device)
        if return_auxiliary_masks:
            batch_size, num_queries = query_features.shape[:2]
            max_spatial = max(
                feat.shape[-2] * feat.shape[-1] for feat in image_features_list
            )
            all_aux_masks = torch.empty(
                self.num_layers,
                self.num_feature_levels,
                batch_size,
                num_queries,
                max_spatial,
                device=query_features.device,
                dtype=query_features.dtype,
            )

        # Process through decoder layers
        for layer_idx in range(self.num_layers):
            layer = self.layers[layer_idx]
            query_features, aux_masks = layer(
                query_features,
                image_features_list,
                mask_features,
                self.pos_query_embed.expand(query_features.shape[0], -1, -1),
                pos_embeddings_list,
                return_auxiliary_masks,
            )

            if aux_masks is not None:
                all_aux_masks[layer_idx] = aux_masks

        # Generate final predictions using prediction heads
        classes, final_masks = self.forward_prediction_heads(
            query_features, mask_features
        )

        return {
            "query_features": query_features,
            "classes": classes,
            "masks": final_masks,
            "auxiliary_masks": all_aux_masks,
        }
