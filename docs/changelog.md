# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added

- **Transformer Decoder Architecture**: Complete implementation of Mask2Former's transformer decoder
  - `DecoderLayer`: Single transformer decoder layer with cross-attention, self-attention, and FFN
  - `TransformerDecoder`: Multi-layer decoder with mask generation and auxiliary outputs
  - Support for multi-scale feature processing with configurable output divisors
- **Attention Mechanisms**: Comprehensive attention module implementations
  - `CrossAttention`: Masked cross-attention for query-to-image feature interaction
  - `SelfAttention`: Standard transformer self-attention for query refinement
  - Multi-head attention support with configurable head counts
  - PyTorch 2.0+ `scaled_dot_product_attention` integration for efficiency
- **Positional Embeddings**: 2D sinusoidal positional encoding
  - `sine_pe_2d`: Generates spatial positional embeddings for transformer inputs
  - Support for arbitrary spatial dimensions and embedding sizes
- **Feed-Forward Networks**: Transformer FFN components
  - `FFN`: Standard transformer feed-forward network with GELU activation
  - `MLP`: Multi-layer perceptron for mask embedding generation
- **Comprehensive Test Suite**: Extensive testing infrastructure
  - Unit tests for all core components with shape validation
  - Gradient flow verification and parameter initialization tests
  - Performance benchmarks for attention mechanisms
  - torch.compile compatibility validation
  - Cross-platform testing (CPU/GPU) with proper synchronization
- **Development Infrastructure**: Enhanced development workflow and tooling
  - Copilot instructions for AI-assisted development
  - MkDocs based documentation with API reference, detailed docstring and navigation
  - Benchmark test markers for performance testing
  - GitHub Actions CI (benchmark tests excluded by default)
  - pre-commit, ruff, pyright for code quality assurance

### Deprecated

- Nothing yet

### Removed

- Nothing yet

### Fixed

- CI pipeline reliability with proper test isolation
- Documentation generation with updated MkDocs configuration

### Security

- Updated dependencies to latest versions with security patches

## [0.1.0] - 2025-10-22
