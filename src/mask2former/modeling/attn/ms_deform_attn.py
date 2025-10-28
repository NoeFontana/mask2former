import torch
import torch.nn as nn
import torch.nn.functional as F


class MSDeformableAttn(nn.Module):
    """Multi-Scale Deformable Attention module based on Deformable DETR.

    Implements proper multi-scale deformable attention with:
    - Learnable sampling offsets for each head/level/point
    - Attention weights to combine sampled features
    - Value projection applied to multi-scale features
    - Proper aggregation across scales and sampling points
    """

    def __init__(
        self, embed_dim: int, num_heads: int, num_levels: int, num_points: int
    ) -> None:
        super().__init__()

        # Validate input dimensions
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.head_dim = embed_dim // num_heads

        # Sampling offset generation: (query) -> (offset_x, offset_y)
        # for each head/level/point
        self.sampling_offsets = nn.Linear(
            embed_dim, num_heads * num_levels * num_points * 2
        )

        # Attention weight generation: (query) -> attention weights
        # for each head/level/point
        self.attention_weights = nn.Linear(
            embed_dim, num_heads * num_levels * num_points
        )  # Value projection applied to input features from each level
        self.value_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)

        # Output projection
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        # Initialize parameters
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize parameters with proper values for deformable attention."""
        # Initialize sampling offsets to small values
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        nn.init.constant_(self.sampling_offsets.bias, 0.0)
        # Initialize attention weights to zero
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        # Initialize value and output projections
        nn.init.xavier_uniform_(self.value_proj.weight)
        if self.value_proj.bias is not None:
            nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

    def _deformable_attention_core(
        self,
        value_features: list[torch.Tensor],
        reference_points: torch.Tensor,
        sampling_offsets: torch.Tensor,
        attention_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Core deformable attention computation.

        Args:
            value_features: Value-projected features for each level.
                List of (N, embed_dim, H, W)
            reference_points: (N, num_query, num_levels, 2)
            sampling_offsets: (N, num_query, num_heads, num_levels, num_points, 2)
            attention_weights: (N, num_query, num_heads, num_levels, num_points)

        Returns:
            torch.Tensor: Aggregated output (N, num_query, embed_dim)
        """
        N, num_query, _ = attention_weights.shape[:3]
        # Reshape value features for multi-head processing
        # (N, C, H, W) -> (N, n_heads, head_dim, H, W) -> (N*n_heads, head_dim, H, W)
        value_features_reshaped = []
        for vf in value_features:
            _, _, H, W = vf.shape
            vf = vf.view(N, self.num_heads, self.head_dim, H, W)
            value_features_reshaped.append(vf.flatten(0, 1))

        # Reshape sampling grid for multi-head processing
        # (N, n_q, n_h, n_l, n_p, 2) -> (N*n_h, n_q, n_l, n_p, 2)
        sampling_offsets = sampling_offsets.permute(0, 2, 1, 3, 4, 5).flatten(0, 1)
        # (N, n_q, n_h, n_l, n_p) -> (N*n_h, n_q, n_l, n_p)
        attention_weights = attention_weights.permute(0, 2, 1, 3, 4).flatten(0, 1)
        # (N, n_q, n_l, 2) -> (N, 1, n_q, n_l, 1, 2) -> (N, n_h, n_q, n_l, 1, 2)
        # -> (N*n_h, n_q, n_l, 1, 2)
        reference_points = (
            reference_points.unsqueeze(1)
            .repeat(1, self.num_heads, 1, 1, 1)
            .flatten(0, 1)
        )

        # Deformable sampling
        sampling_locations = (
            reference_points.unsqueeze(-2) + sampling_offsets
        )  # (N*n_h, n_q, n_l, n_p, 2)

        # Normalize to [-1, 1] for grid_sample
        sampling_grid = 2.0 * sampling_locations - 1.0

        # Sample features: for each level, sample from value features
        sampled_features = []
        for level, value_feat in enumerate(value_features_reshaped):
            # (N*n_h, n_q, n_p, 2)
            level_sampling_grid = sampling_grid[:, :, level, :, :]

            # (N*n_h, h_dim, H, W), (N*n_h, n_q, n_p, 2) -> (N*n_h, h_dim, n_q, n_p)
            sampled = F.grid_sample(
                value_feat,
                level_sampling_grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            sampled_features.append(sampled)

        # (N*n_h, h_dim, n_q, n_l, n_p)
        sampled_features = torch.stack(sampled_features, dim=-2)

        # Apply attention weights
        # (N*n_h, n_q, n_l, n_p) -> (N*n_h, 1, n_q, n_l, n_p)
        attention_weights = F.softmax(attention_weights.flatten(2), dim=-1).view_as(
            attention_weights
        )
        attention_weights = torch.nan_to_num(attention_weights)
        attention_weights = attention_weights.unsqueeze(1)
        # (N*n_h, h_dim, n_q, n_l, n_p) * (N*n_h, 1, n_q, n_l, n_p)
        # -> (N*n_h, h_dim, n_q, n_l, n_p)
        weighted_features = sampled_features * attention_weights

        # Sum over levels and points
        # (N*n_h, h_dim, n_q)
        output = weighted_features.sum(dim=[-1, -2])

        # Reshape back to (N, num_query, embed_dim)
        # (N*n_h, h_dim, n_q) -> (N, n_h, h_dim, n_q) -> (N, n_q, n_h, h_dim)
        # -> (N, n_q, C)
        output = (
            output.view(N, self.num_heads, self.head_dim, num_query)
            .permute(0, 3, 1, 2)
            .flatten(2)
        )

        return output

    def forward(
        self,
        query: torch.Tensor,
        multi_scale_features: list[torch.Tensor],
        reference_points: torch.Tensor,
        padding_mask: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Multi-scale deformable attention forward pass.

        Args:
            query (torch.Tensor): Query tensor from decoder.
                Shape: (N, num_query, embed_dim)
            multi_scale_features (list[torch.Tensor]): Multi-scale feature maps.
                List of tensors with shape (N, embed_dim, H_l, W_l) for level l
            reference_points (torch.Tensor): Reference points in [0, 1] coordinates.
                Shape: (N, num_query, 2) or (N, num_query, num_levels, 2)
            padding_mask (list[torch.Tensor], optional): Padding masks for each
                level. List of boolean tensors with shape (N, H_l, W_l) where
                True indicates valid positions, False indicates padding.

        Returns:
            torch.Tensor: Output tensor.
                Shape: (N, num_query, embed_dim)
        """
        N, num_query, _ = query.shape

        # Generate sampling offsets
        # Shape: (N, n_q, n_h, n_l, n_p, 2)
        sampling_offsets = self.sampling_offsets(query).view(
            N, num_query, self.num_heads, self.num_levels, self.num_points, 2
        )

        # Generate attention weights (logits)
        # Shape: (N, n_q, n_h, n_l, n_p)
        attention_weights = self.attention_weights(query).view(
            N, num_query, self.num_heads, self.num_levels, self.num_points
        )

        # Prepare reference points: expand to per-level if needed
        if reference_points.dim() == 3:  # (N, num_query, 2)
            reference_points = reference_points.unsqueeze(2).expand(
                N, num_query, self.num_levels, 2
            )

        # Apply value projection to all feature levels
        value_features = [self.value_proj(feat) for feat in multi_scale_features]

        # Mask attention weights for padded regions
        if padding_mask is not None:
            # (N, n_q, n_h, n_l, n_p, 2)
            sampling_locations = (
                reference_points.unsqueeze(2).unsqueeze(4) + sampling_offsets
            )
            for i, mask in enumerate(padding_mask):
                if i >= self.num_levels:
                    break
                # (N, H, W) -> (N, 1, H, W) for grid_sample
                mask = mask.unsqueeze(1).to(sampling_locations.dtype)
                # (N, n_q, n_h, n_p, 2)
                level_sampling_locs = sampling_locations[:, :, :, i, :, :]
                # Normalize to [-1, 1]
                level_sampling_locs = 2.0 * level_sampling_locs - 1.0
                # (N, n_q, n_h, n_p) -> (N, n_q * n_h * n_p)
                is_valid = (
                    (level_sampling_locs[..., 0] >= -1.0)
                    & (level_sampling_locs[..., 0] <= 1.0)
                    & (level_sampling_locs[..., 1] >= -1.0)
                    & (level_sampling_locs[..., 1] <= 1.0)
                )
                # (N*n_h, 1, H, W), (N*n_h, n_q, n_p, 2) -> (N*n_h, 1, n_q, n_p)
                grid = level_sampling_locs.permute(0, 2, 1, 3, 4).flatten(0, 1)
                repeated_mask = mask.repeat(self.num_heads, 1, 1, 1)
                sampled_mask = F.grid_sample(
                    repeated_mask,
                    grid,
                    mode="nearest",
                    padding_mode="zeros",
                    align_corners=False,
                )  # (N*n_h, 1, n_q, n_p)
                # (N, n_q, n_h, n_p)
                sampled_mask = sampled_mask.view(
                    N, self.num_heads, num_query, self.num_points
                ).permute(0, 2, 1, 3)
                # Mask attention weights
                is_valid &= sampled_mask.bool()
                attention_weights[:, :, :, i, :].masked_fill_(~is_valid, -torch.inf)

        # Sample features using deformable attention
        output = self._deformable_attention_core(
            value_features,
            reference_points,
            sampling_offsets,
            attention_weights,
        )

        # Final output projection
        return self.output_proj(output)
