# API Reference

This page contains the API reference for the `mask2former` package.

## Package Overview

::: mask2former

## Core Modeling Components

### Backbone

The backbone module provides feature extraction using timm models.

::: mask2former.modeling.backbone
options:
heading_level: 3

### Decoder

The decoder module implements the Mask2Former transformer decoder architecture.

::: mask2former.modeling.decoder
options:
heading_level: 3

### Positional Embeddings

Utilities for generating positional embeddings.

::: mask2former.modeling.pe
options:
heading_level: 3

## Common Components

### Classification and Segmentation Heads

Components for generating final predictions from query embeddings.

::: mask2former.modeling.common.head
options:
heading_level: 3

### Feed-Forward Networks

Feed-forward network components used in the transformer architecture.

::: mask2former.modeling.common.ffn
options:
heading_level: 3
