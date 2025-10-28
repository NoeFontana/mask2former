# AGENT Instructions for Mask2Former

This guide provides instructions for AI agents to effectively contribute to the Mask2Former project.

## Project Overview

This is a PyTorch reimplementation of [Mask2Former](https://arxiv.org/pdf/2112.01527) for image segmentation, built around **timm backbones** and a transformer decoder architecture. The project is in **alpha status**, with core functionality under active development.

## Core Workflows

### Development Commands (via Makefile)

- `make check`: Run the complete quality pipeline (lint + format-check + type-check + test). **Always run this before submitting.**
- `make test`: Run the test suite with pytest.
- `make test-cov`: Run tests and generate an HTML coverage report in `htmlcov/`.
- `make lint`: Lint and auto-fix issues with ruff.
- `make format`: Format code with ruff.
- `make type-check`: Run static type checking with pyright.
- `make docs`: Serve documentation locally with auto-reload.

### Testing Conventions

- Tests are located in the `tests/` directory and use **pytest**.
- The standard test fixture is `sample_image`, which provides a realistic image for processing.
- Test pattern:
  1. Input validation.
  2. Model forward pass.
  3. Assertions on output shapes and types.

## Architecture and Coding Conventions

### Core Components (`src/mask2former/modeling/`)

- **`backbone.py`**: A thin wrapper around `timm` models for multi-scale feature extraction.
- **`pixel_decoder/` and `transformer_decoder/`**: These components form the core of the segmentation model. The transformer decoder uses custom attention mechanisms.
- **`common/`**: Shared components like FFN and MLP layers.

### Data Flow

1.  Image -> `timm` backbone -> Multi-scale features
2.  Features + Positional Embeddings -> Transformer Decoder
3.  Query Features -> Mask Embeddings -> Segmentation Masks

### Coding Conventions

- **Logging**: Use the `logging` module instead of `print()` for any debug or informational output.
- **Tensor Shapes**: Always document tensor shapes in docstrings.
  ```python
  Args:
      x (torch.Tensor): Input tensor.
          Shape: (batch_size, channels, height, width)
  ```
- **Attention Modules**: Follow the established pattern of separate Q/K/V projectors and use `torch.nn.functional.scaled_dot_product_attention`.

## Key Files for Context

- **`pyproject.toml`**: Contains all project configuration, dependencies, and tool settings.
- **`Makefile`**: Defines all development commands and workflows.
- **`tests/conftest.py`**: Shared test fixtures and configurations.
- **`src/mask2former/modeling/transformer_decoder/decoder.py`**: The core transformer implementation.
- **`src/mask2former/modeling/backbone.py`**: The `timm` backbone wrapper.

## Dependencies

- **Python 3.11+**
- **PyTorch 2.9+**
- **timm 1.0.20+**
- Use `uv sync --all-extras` to install and manage dependencies.
