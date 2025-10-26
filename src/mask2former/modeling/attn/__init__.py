"""Attention modules for Mask2Former transformer decoder."""

from mask2former.modeling.attn.cross_attn import CrossAttention
from mask2former.modeling.attn.self_attn import SelfAttention

__all__ = [
    "CrossAttention",
    "SelfAttention",
]
