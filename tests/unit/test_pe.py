"""Unit tests for positional encoding functions."""

import pytest
import torch

from mask2former.modeling.pe import sine_pe_2d


class TestSinePE2D:
    """Test suite for sine_pe_2d function."""

    def test_basic_functionality(self):
        """Test basic functionality with standard parameters."""
        pe = sine_pe_2d(embedding_dim=256, height=32, width=32)

        assert pe.shape == (256, 32, 32)
        assert pe.dtype == torch.float32
        assert torch.isfinite(pe).all()

    def test_small_dimensions(self):
        """Test with minimal valid dimensions."""
        pe = sine_pe_2d(embedding_dim=4, height=2, width=2)

        assert pe.shape == (4, 2, 2)
        assert pe.dtype == torch.float32

    def test_large_dimensions(self):
        """Test with larger dimensions."""
        pe = sine_pe_2d(embedding_dim=512, height=64, width=128)

        assert pe.shape == (512, 64, 128)
        assert pe.dtype == torch.float32

    def test_embedding_dim_not_divisible_by_4(self):
        """Test that ValueError is raised when embedding_dim is not divisible by 4."""
        with pytest.raises(
            ValueError, match="Embedding dimension must be divisible by 4"
        ):
            sine_pe_2d(embedding_dim=3, height=32, width=32)

        with pytest.raises(
            ValueError, match="Embedding dimension must be divisible by 4"
        ):
            sine_pe_2d(embedding_dim=10, height=32, width=32)

    def test_output_range(self):
        """Test that output values are within expected range [-1, 1]."""
        pe = sine_pe_2d(embedding_dim=256, height=32, width=32)

        assert pe.min() >= -1.0
        assert pe.max() <= 1.0

    def test_deterministic_output(self):
        """Test that function produces deterministic output."""
        pe1 = sine_pe_2d(embedding_dim=256, height=32, width=32)
        pe2 = sine_pe_2d(embedding_dim=256, height=32, width=32)

        assert torch.equal(pe1, pe2)
