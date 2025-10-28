import logging

import pytest
import torch

from mask2former.modeling.attn import MSDeformableAttn

logger = logging.getLogger(__name__)


@pytest.fixture(scope="class")
def standard_ms_deform_attn() -> MSDeformableAttn:
    """Fixture providing a standard MSDeformableAttn for testing."""
    return MSDeformableAttn(embed_dim=256, num_heads=8, num_levels=3, num_points=4)


@pytest.fixture(scope="class")
def small_ms_deform_attn() -> MSDeformableAttn:
    """Fixture providing a small MSDeformableAttn for quick tests."""
    return MSDeformableAttn(embed_dim=64, num_heads=2, num_levels=2, num_points=2)


class TestMSDeformableAttn:
    """Tests for the MSDeformableAttn class."""

    def test_initialization_error(self) -> None:
        """Test initialization with invalid dimensions raises ValueError."""
        with pytest.raises(
            ValueError, match="embed_dim .* must be divisible by num_heads"
        ):
            MSDeformableAttn(embed_dim=100, num_heads=7, num_levels=2, num_points=2)

    def test_initialization(self, standard_ms_deform_attn: MSDeformableAttn) -> None:
        """Test module initialization with correct parameters."""
        ms_deform_attn = standard_ms_deform_attn
        assert hasattr(ms_deform_attn, "sampling_offsets")
        assert hasattr(ms_deform_attn, "attention_weights")
        assert hasattr(ms_deform_attn, "value_proj")

        # Check dimensions
        expected_offset_features = 3 * 8 * 4 * 2  # n_levels * n_heads * n_points * 2
        assert ms_deform_attn.sampling_offsets.out_features == expected_offset_features

        expected_attn_features = 3 * 8 * 4  # n_levels * n_heads * n_points
        assert ms_deform_attn.attention_weights.out_features == expected_attn_features

    def test_forward_pass_shape(self, small_ms_deform_attn: MSDeformableAttn) -> None:
        """Test forward pass produces correct output shape."""
        attention = small_ms_deform_attn
        batch_size, num_query = 2, 10

        # Create test inputs
        query = torch.randn(batch_size, num_query, 64)

        # Multi-scale features with different spatial sizes
        multi_scale_features = [
            torch.randn(batch_size, 64, 32, 32),  # Level 0
            torch.randn(batch_size, 64, 16, 16),  # Level 1
        ]

        # Reference points in [0, 1] coordinates
        reference_points = torch.rand(batch_size, num_query, 2)

        # Forward pass
        output = attention(query, multi_scale_features, reference_points)

        # Check output shape
        assert output.shape == (batch_size, num_query, 64)
        assert output.dtype == torch.float32

    def test_forward_with_per_level_reference_points(
        self, small_ms_deform_attn: MSDeformableAttn
    ) -> None:
        """Test forward pass with per-level reference points."""
        attention = small_ms_deform_attn
        batch_size, num_query = 2, 10

        # Create test inputs
        query = torch.randn(batch_size, num_query, 64)

        multi_scale_features = [
            torch.randn(batch_size, 64, 32, 32),
            torch.randn(batch_size, 64, 16, 16),
        ]

        # Per-level reference points: (N, num_query, num_levels, 2)
        reference_points = torch.rand(batch_size, num_query, 2, 2)

        # Forward pass should work with per-level reference points
        output = attention(query, multi_scale_features, reference_points)
        assert output.shape == (batch_size, num_query, 64)

    def test_gradient_flow(self, small_ms_deform_attn: MSDeformableAttn) -> None:
        """Test that gradients flow through the module."""
        attention = small_ms_deform_attn
        batch_size, num_query = 1, 5

        # Create test inputs with gradient tracking
        query = torch.randn(batch_size, num_query, 64, requires_grad=True)

        multi_scale_features = [
            torch.randn(batch_size, 64, 16, 16, requires_grad=True),
            torch.randn(batch_size, 64, 8, 8, requires_grad=True),
        ]

        reference_points = torch.rand(batch_size, num_query, 2, requires_grad=True)

        # Forward pass
        output = attention(query, multi_scale_features, reference_points)

        # Backward pass
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        assert query.grad is not None
        assert multi_scale_features[0].grad is not None
        assert multi_scale_features[1].grad is not None
        assert reference_points.grad is not None

    def test_edge_cases(self, standard_ms_deform_attn: MSDeformableAttn) -> None:
        """Test edge cases and boundary conditions."""
        attention = standard_ms_deform_attn

        # Test with single query
        query = torch.randn(1, 1, 256)
        multi_scale_features = [
            torch.randn(1, 256, 8, 8),
            torch.randn(1, 256, 4, 4),
            torch.randn(1, 256, 2, 2),
        ]
        reference_points = torch.rand(1, 1, 2)

        output = attention(query, multi_scale_features, reference_points)
        assert output.shape == (1, 1, 256)

        # Test with boundary reference points
        reference_points_boundary = torch.tensor(
            [[[0.0, 0.0], [1.0, 1.0]]], dtype=torch.float32
        )
        query_boundary = torch.randn(1, 2, 256)

        output_boundary = attention(
            query_boundary,
            multi_scale_features,
            reference_points_boundary,
        )
        assert output_boundary.shape == (1, 2, 256)

    def test_deterministic_output(self, small_ms_deform_attn: MSDeformableAttn) -> None:
        """Test that the same input produces the same output (deterministic)."""
        attention = small_ms_deform_attn
        attention.eval()  # Set to evaluation mode

        # Create consistent test inputs
        torch.manual_seed(42)
        query = torch.randn(1, 3, 64)
        multi_scale_features = [
            torch.randn(1, 64, 8, 8),
            torch.randn(1, 64, 4, 4),
        ]
        reference_points = torch.rand(1, 3, 2)

        # First forward pass
        output1 = attention(query, multi_scale_features, reference_points)

        # Second forward pass with same inputs
        output2 = attention(query, multi_scale_features, reference_points)

        # Outputs should be identical
        assert torch.allclose(output1, output2, atol=1e-6)

    def test_torch_compile_compatibility(
        self, small_ms_deform_attn: MSDeformableAttn
    ) -> None:
        """Test that the module is compatible with torch.compile."""
        attention = small_ms_deform_attn
        attention.eval()

        # Compile the module
        compiled_attention = torch.compile(attention, fullgraph=True)

        # Create test inputs
        batch_size, num_query = 2, 5
        query = torch.randn(batch_size, num_query, 64)
        multi_scale_features = [
            torch.randn(batch_size, 64, 16, 16),
            torch.randn(batch_size, 64, 8, 8),
        ]
        reference_points = torch.rand(batch_size, num_query, 2)

        # Get outputs from both original and compiled modules
        with torch.no_grad():
            original_output = attention(query, multi_scale_features, reference_points)
            compiled_output = compiled_attention(
                query, multi_scale_features, reference_points
            )

        # Outputs should be identical
        assert torch.allclose(original_output, compiled_output, atol=1e-5)
        assert original_output.shape == compiled_output.shape
        logger.info("torch.compile compatibility verified")

    def test_torch_export_compatibility(
        self, small_ms_deform_attn: MSDeformableAttn
    ) -> None:
        """Test that the module is compatible with torch.export."""
        attention = small_ms_deform_attn
        attention.eval()

        # Create example inputs for export
        batch_size, num_query = 1, 3
        query = torch.randn(batch_size, num_query, 64)
        multi_scale_features = [
            torch.randn(batch_size, 64, 8, 8),
            torch.randn(batch_size, 64, 4, 4),
        ]
        reference_points = torch.rand(batch_size, num_query, 2)

        # Export the module
        exported_program = torch.export.export(
            attention,
            args=(query, multi_scale_features, reference_points),
            strict=True,
        )

        # Verify the exported program works
        with torch.no_grad():
            original_output = attention(query, multi_scale_features, reference_points)
            exported_output = exported_program.module()(
                query, multi_scale_features, reference_points
            )

        # Outputs should be identical
        assert torch.allclose(original_output, exported_output, atol=1e-5)
        assert original_output.shape == exported_output.shape
        assert exported_program is not None
        logger.info("torch.export compatibility verified")


class TestAttnProjector:
    """Tests for the AttnProjector helper class."""

    def test_attention_projector_shape(
        self, standard_ms_deform_attn: MSDeformableAttn
    ) -> None:
        """Test that attention projector produces correct shapes."""
        projector = standard_ms_deform_attn.attention_weights
        n, n_q, n_h, n_l, n_p = (
            2,
            10,
            standard_ms_deform_attn.num_heads,
            standard_ms_deform_attn.num_levels,
            standard_ms_deform_attn.num_points,
        )

        query = torch.randn(n, n_q, 256)
        attention_logits = projector(query).view(n, n_q, n_h, n_l, n_p)

        # Should output (N, num_query, num_heads, num_levels, num_points)
        expected_shape = (n, n_q, n_h, n_l, n_p)
        assert attention_logits.shape == expected_shape

        # Should be raw logits (not normalized), but can be normalized with softmax
        attention_weights = torch.softmax(attention_logits.flatten(-2), dim=-1)
        assert torch.allclose(
            attention_weights.sum(dim=-1),
            torch.ones(n, n_q, n_h),
            atol=1e-6,
        )

    def test_attention_weights_range(
        self, small_ms_deform_attn: MSDeformableAttn
    ) -> None:
        """Test that attention logits can be normalized to valid range [0, 1]."""
        projector = small_ms_deform_attn.attention_weights
        n, n_q, n_h, n_l, n_p = (
            1,
            5,
            small_ms_deform_attn.num_heads,
            small_ms_deform_attn.num_levels,
            small_ms_deform_attn.num_points,
        )

        query = torch.randn(n, n_q, 64)
        attention_logits = projector(query).view(n, n_q, n_h, n_l, n_p)

        # Logits can be any real number, but when normalized should be in [0, 1]
        attention_weights = torch.softmax(attention_logits, dim=-1)
        assert torch.all(attention_weights >= 0)
        assert torch.all(attention_weights <= 1)


class TestPaddingMaskSupport:
    """Tests for padding mask functionality in MSDeformableAttn."""

    def test_forward_with_padding_mask(
        self, small_ms_deform_attn: MSDeformableAttn
    ) -> None:
        """Test forward pass with padding masks produces correct output shape."""
        attention = small_ms_deform_attn
        batch_size, num_query = 2, 10

        # Create test inputs
        query = torch.randn(batch_size, num_query, 64)

        # Multi-scale features with different spatial sizes
        multi_scale_features = [
            torch.randn(batch_size, 64, 32, 32),  # Level 0
            torch.randn(batch_size, 64, 16, 16),  # Level 1
        ]

        # Reference points in [0, 1] coordinates
        reference_points = torch.rand(batch_size, num_query, 2)

        # Create padding masks for each level (True = valid, False = padded)
        padding_masks = [
            torch.ones(batch_size, 32, 32, dtype=torch.bool),  # Level 0
            torch.ones(batch_size, 16, 16, dtype=torch.bool),  # Level 1
        ]

        # Forward pass with padding masks
        output = attention(query, multi_scale_features, reference_points, padding_masks)

        # Check output shape remains unchanged
        assert output.shape == (batch_size, num_query, 64)
        assert output.dtype == torch.float32

    def test_padding_mask_masking_behavior(
        self, small_ms_deform_attn: MSDeformableAttn
    ) -> None:
        """Test that padding mask properly masks out padded regions."""
        attention = small_ms_deform_attn
        batch_size, num_query = 1, 3

        query = torch.randn(batch_size, num_query, 64)

        # Two levels for small_ms_deform_attn
        multi_scale_features = [
            torch.randn(batch_size, 64, 8, 8),
            torch.randn(batch_size, 64, 4, 4),
        ]

        # Reference points pointing to specific locations
        reference_points = torch.tensor(
            [[[0.25, 0.25], [0.75, 0.75], [0.5, 0.5]]], dtype=torch.float32
        )

        # Create padding mask that masks out the bottom-right quadrant of level 0
        padding_mask_level0 = torch.ones(batch_size, 8, 8, dtype=torch.bool)
        padding_mask_level0[:, 4:, 4:] = False  # Mask bottom-right quadrant
        padding_mask_level1 = torch.ones(batch_size, 4, 4, dtype=torch.bool)

        padding_masks = [padding_mask_level0, padding_mask_level1]

        # Forward pass without mask
        output_no_mask = attention(query, multi_scale_features, reference_points, None)

        # Forward pass with mask
        output_with_mask = attention(
            query,
            multi_scale_features,
            reference_points,
            padding_masks,
        )

        # Outputs should be different due to masking
        assert not torch.allclose(output_no_mask, output_with_mask, atol=1e-6)

        # Both outputs should have valid values (no NaN/Inf)
        assert torch.all(torch.isfinite(output_no_mask))
        assert torch.all(torch.isfinite(output_with_mask))

    def test_padding_mask_gradient_flow(
        self, small_ms_deform_attn: MSDeformableAttn
    ) -> None:
        """Test that gradients flow correctly through padding mask operations."""
        attention = small_ms_deform_attn
        batch_size, num_query = 1, 2

        # Create test inputs with gradient tracking
        query = torch.randn(batch_size, num_query, 64, requires_grad=True)
        multi_scale_features = [
            torch.randn(batch_size, 64, 8, 8, requires_grad=True),
            torch.randn(batch_size, 64, 4, 4, requires_grad=True),
        ]
        reference_points = torch.rand(batch_size, num_query, 2, requires_grad=True)

        # Create padding masks
        padding_masks = [
            torch.ones(batch_size, 8, 8, dtype=torch.bool),
            torch.ones(batch_size, 4, 4, dtype=torch.bool),
        ]
        # Mask some regions
        padding_masks[0][:, :2, :] = False  # Mask top rows of first level
        padding_masks[1][:, 2:, 2:] = False  # Mask bottom-right of second level

        # Forward pass
        output = attention(query, multi_scale_features, reference_points, padding_masks)

        # Backward pass
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        assert query.grad is not None
        assert multi_scale_features[0].grad is not None
        assert multi_scale_features[1].grad is not None
        assert reference_points.grad is not None

        # Check gradients are finite
        assert torch.all(torch.isfinite(query.grad))
        assert torch.all(torch.isfinite(multi_scale_features[0].grad))
        assert torch.all(torch.isfinite(multi_scale_features[1].grad))
        assert torch.all(torch.isfinite(reference_points.grad))

    def test_partial_padding_mask_list(
        self, small_ms_deform_attn: MSDeformableAttn
    ) -> None:
        """Test behavior when padding mask list has fewer elements than features."""
        attention = small_ms_deform_attn
        batch_size, num_query = 1, 2

        query = torch.randn(batch_size, num_query, 64)
        multi_scale_features = [
            torch.randn(batch_size, 64, 8, 8),
            torch.randn(batch_size, 64, 4, 4),  # This level won't have a mask
        ]
        reference_points = torch.rand(batch_size, num_query, 2)

        # Only provide mask for first level
        padding_masks = [
            torch.ones(batch_size, 8, 8, dtype=torch.bool),
        ]

        # Forward pass should work without error
        output = attention(query, multi_scale_features, reference_points, padding_masks)

        assert output.shape == (batch_size, num_query, 64)
        assert torch.all(torch.isfinite(output))

    def test_empty_padding_mask_list(
        self, small_ms_deform_attn: MSDeformableAttn
    ) -> None:
        """Test behavior with empty padding mask list."""
        attention = small_ms_deform_attn
        batch_size, num_query = 1, 2

        query = torch.randn(batch_size, num_query, 64)
        multi_scale_features = [
            torch.randn(batch_size, 64, 8, 8),
            torch.randn(batch_size, 64, 4, 4),
        ]
        reference_points = torch.rand(batch_size, num_query, 2)

        # Empty padding mask list
        padding_masks = []

        # Forward pass should work like no mask provided
        output = attention(query, multi_scale_features, reference_points, padding_masks)

        assert output.shape == (batch_size, num_query, 64)
        assert torch.all(torch.isfinite(output))

    def test_all_masked_region(self, small_ms_deform_attn: MSDeformableAttn) -> None:
        """Test behavior when entire regions are masked out."""
        attention = small_ms_deform_attn
        batch_size, num_query = 1, 2

        query = torch.randn(batch_size, num_query, 64)
        multi_scale_features = [
            torch.randn(batch_size, 64, 4, 4),
            torch.randn(batch_size, 64, 2, 2),
        ]
        reference_points = torch.rand(batch_size, num_query, 2)

        # Completely masked padding mask
        padding_mask = [
            torch.zeros(batch_size, 4, 4, dtype=torch.bool),
            torch.zeros(batch_size, 2, 2, dtype=torch.bool),
        ]

        # Forward pass with completely masked features
        output = attention(
            query,
            multi_scale_features,
            reference_points,
            padding_mask,
        )

        # Output should be finite (zeros due to masking)
        assert output.shape == (batch_size, num_query, 64)
        assert torch.all(torch.isfinite(output))

        # Since everything is masked, output should be close to zero
        # (after value and output projections might add small non-zero values)
        assert torch.all(torch.abs(output) < 1e-5)  # Reasonable threshold
