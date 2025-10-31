# Mask2Former

[![Status](https://img.shields.io/badge/status-alpha-orange.svg)](https://github.com/NoeFontana/mask2former)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/NoeFontana/mask2former/actions/workflows/ci.yml/badge.svg)](https://github.com/NoeFontana/mask2former/actions)

_**Alpha status**: Core functionality is under active development._

A [PyTorch](https://pytorch.org/projects/pytorch/) reimplementation of [Mask2Former](https://arxiv.org/pdf/2112.01527) using timm backbones.

## Quick Start

```bash
# Clone and setup
git clone https://github.com/NoeFontana/mask2former.git
cd mask2former

# Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dev dependencies
uv sync --all-groups

# Set up pre-commit hooks
make pre-commit
```

## Core Workflows

### Development Commands

```bash
# Run all quality checks
make check                   # lint + format-check + type-check + test

# Individual checks
make test                    # Run test suite
make test-cov                # Run tests with coverage report
make lint                    # Lint and auto-fix issues
make format                  # Format code
make type-check              # Type checking with pyright
```

### Documentation

```bash
make docs                    # Serve docs locally
make docs-build              # Build static docs
```

### Maintenance

```bash
make clean                   # Clean build artifacts and cache
make update                  # Update all dependencies
```

## Project Structure

```
src/mask2former/               # Main package
├── modeling/                  # Core model components
│   ├── attn/                  # Attention mechanisms
│   ├── backbone.py            # timm backbone wrapper
│   ├── common/                # Shared components
│   ├── pe.py                  # Positional embedings
│   ├── pixel_decoder/         # Pixel decoder
│   └── transformer_decoder/   # Transformer decoder
tests/                         # Test suite
docs/                          # Documentation
Makefile                       # Development commands
pyproject.toml                 # Project configuration
```

## Contributing

1. Fork and create a feature branch
2. Make changes with tests
3. Run `make check` to ensure quality
4. Submit a pull request

**Requirements:** Python 3.11+, comprehensive tests, type hints, and documentation updates.

## License

MIT License - see [LICENSE](LICENSE) for details.
