from typing import cast

import torch
from torch import nn

from mask2former.modeling.attn import CrossAttention, SelfAttention
from mask2former.modeling.common.ffn import FFN, MLP
from mask2former.modeling.pe import sine_pe_2d


def generate_mask_logits(
    query_features: torch.Tensor,
    pixel_features: torch.Tensor,
    mask_embedder: MLP,
) -> torch.Tensor:
    """Convert query features to mask logits at pixel_features resolution.

    Args:
        query_features (torch.Tensor): Query features tensor.
            Shape: (batch_size, num_queries, embedding_dim)
        pixel_features (torch.Tensor): Pixel features tensor.
            Shape: (batch_size, embedding_dim, height, width)
        mask_embedder (MLP): MLP module to convert query features to mask embeddings.

    Returns:
        torch.Tensor: Mask logits tensor at pixel_features resolution.
            Shape: (batch_size, num_queries, height, width)
    """
    mask_embed = mask_embedder(query_features)
    # Mask logits are obtained via a dot product over the embedding dimension
    outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, pixel_features)

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

        self.cross_attention = CrossAttention(embedding_dim, num_head)
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
            + self.cross_attention(
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


class TransformerDecoder(nn.Module):
    """Transformer Decoder for Mask2Former.

    It consists of multiple decoder layers, each processing multiple feature levels.
    """

    def __init__(
        self,
        num_layers: int,
        embedding_dim: int,
        num_query: int,
        num_head: int,
        hidden_dim: int,
        num_feature_levels: int = 3,
        mask_embedder_hidden_dim: int = 256,
        input_size: tuple[int, int] = (512, 512),
        output_divisors: tuple[int, ...] = (32, 16, 8),
    ) -> None:
        super().__init__()

        if len(output_divisors) != num_feature_levels:
            raise ValueError(
                f"The length of output_divisors ({len(output_divisors)}) must match "
                f"num_feature_levels ({num_feature_levels})."
            )

        self.num_layers = num_layers
        self.num_feature_levels = num_feature_levels

        self.mask_embedder = MLP(embedding_dim, hidden_dim=mask_embedder_hidden_dim)

        self.initial_queries = nn.Parameter(torch.randn(1, num_query, embedding_dim))
        self.pos_query_embed = nn.Parameter(torch.randn(1, num_query, embedding_dim))

        self.decoder_norm = nn.LayerNorm(embedding_dim)

        # Create a nested ModuleList for better structural representation
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        DecoderLayer(embedding_dim, num_head, hidden_dim)
                        for _ in range(num_feature_levels)
                    ]
                )
                for _ in range(num_layers)
            ]
        )

        # Input configuration for positional embeddings
        self.input_height, self.input_width = input_size
        self.output_divisors = output_divisors

        # Register positional embeddings as buffers and create static list
        for i, divisor in enumerate(self.output_divisors):
            ape = sine_pe_2d(
                embedding_dim,
                self.input_height // divisor,
                self.input_width // divisor,
                device=self.initial_queries.device,
            )
            # Reshape for use in attention: (1, H*W, C)
            ape = ape.permute(1, 2, 0).view(1, -1, embedding_dim)
            self.register_buffer(f"ape_{i}", ape)

        # Initialize parameters
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize learnable parameters."""
        nn.init.xavier_uniform_(self.initial_queries)
        nn.init.xavier_uniform_(self.pos_query_embed)

    def forward_prediction_heads(
        self,
        query_features: torch.Tensor,
        mask_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward prediction heads.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple containing:
                - decoder_output (torch.Tensor): Normalized query features.
                    Shape: (batch_size, num_queries, embedding_dim)
                - outputs_mask (torch.Tensor): Mask logits at mask_features resolution.
                    Shape: (batch_size, num_queries, height, width)
        """
        decoder_output = self.decoder_norm(query_features)

        outputs_mask = generate_mask_logits(
            decoder_output,
            mask_features,
            self.mask_embedder,
        )

        return decoder_output, outputs_mask

    def forward(
        self,
        image_features_list: list[torch.Tensor],
        mask_features: torch.Tensor,
        return_auxiliary_masks: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Forward pass optimized for torch.compile and torch.export."""
        device, dtype = self.initial_queries.device, self.initial_queries.dtype

        batch_size = mask_features.shape[0]
        query_features = self.initial_queries.expand(batch_size, -1, -1)
        pos_query = self.pos_query_embed.expand(batch_size, -1, -1)

        # Create a static list of positional embeddings from buffers.
        # This is torch.compile-friendly as the list is created from static attributes.
        pos_embeddings_list = [
            getattr(self, f"ape_{i}") for i in range(self.num_feature_levels)
        ]
        all_aux_masks: list[torch.Tensor] = []

        for layer in self.layers:
            layer = cast(nn.ModuleList, layer)
            layer_aux_masks: list[torch.Tensor] = []
            for level_idx, decoder_layer in enumerate(layer):
                image_features = image_features_list[level_idx]
                pos_image_embeddings = pos_embeddings_list[level_idx]

                # Generate mask logits at mask_features resolution (not image_features)
                mask_logits = generate_mask_logits(
                    query_features,
                    mask_features,
                    self.mask_embedder,
                )

                image_height, image_width = image_features.shape[-2:]

                if return_auxiliary_masks:
                    layer_aux_masks.append(mask_logits)

                attention_mask = generate_attention_mask(
                    mask_logits, (image_height, image_width)
                )

                query_features = decoder_layer(
                    query_features,
                    image_features,
                    attention_mask,
                    pos_query,
                    pos_image_embeddings,
                )

            if return_auxiliary_masks and layer_aux_masks:
                all_aux_masks.append(torch.stack(layer_aux_masks, dim=0))

        final_features, final_masks = self.forward_prediction_heads(
            query_features, mask_features
        )

        # Create auxiliary masks tensor with better torch.compile compatibility
        if return_auxiliary_masks and all_aux_masks:
            aux_masks_tensor = torch.stack(all_aux_masks, dim=0)
        else:
            aux_masks_tensor = torch.empty(0, device=device, dtype=dtype)

        return {
            "query_features": final_features,
            "masks": final_masks,
            "auxiliary_masks": aux_masks_tensor,
        }
