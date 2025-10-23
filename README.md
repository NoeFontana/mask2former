# Python Template

[![CI](https://github.com/noe-fontana/mask2former/workflows/CI/badge.svg)](https://github.com/noe-fontana/mask2former/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with pyright](https://microsoft.github.io/pyright/img/pyright_badge.svg)](https://microsoft.github.io/pyright/)

A [PyTorch](https://pytorch.org/projects/pytorch/) reimplementation of [Mask2Former](https://arxiv.org/pdf/2112.01527) using timm backbones.

## Requirements

- Python 3.11 or higher
- [uv](https://docs.astral.sh/uv/) for dependency management

## Installation

### Using uv (recommended)

```bash
# Clone the repository
git clone https://github.com/NoeFontana/mask2former.git
cd mask2former

# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a virtual environment and install dependencies
uv sync

# Install development dependencies
uv sync --all-extras
```

## Usage

### Development

```bash
# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov

# Lint and format code
uv run ruff check .          # Check for issues
uv run ruff check . --fix    # Fix auto-fixable issues
uv run ruff format .         # Format code

# Type checking
uv run pyright

# Run all checks (lint, format, type check, test)
make check  # If using the provided Makefile
```

### Pre-commit hooks

Set up pre-commit hooks to automatically run code quality checks:

```bash
uv run pre-commit install

# Run hooks manually
uv run pre-commit run --all-files
```

### Documentation

Build and serve documentation locally:

```bash
# Install docs dependencies
uv sync --extra docs

# Serve documentation locally
uv run mkdocs serve

# Build documentation
uv run mkdocs build
```

## Project Structure

```
mask2former/
├── .github/
│   └── workflows/          # GitHub Actions CI/CD
├── docs/                   # Documentation source
├── src/
│   └── mask2former/    # Main package source code
│       ├── __init__.py
│       └── ...
├── tests/                  # Test suite
│   ├── __init__.py
│   └── ...
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── Makefile               # Development shortcuts
├── pyproject.toml         # Project configuration
└── README.md
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run the test suite (`uv run pytest`)
5. Run code quality checks (`uv run ruff check . && uv run ruff format . && uv run pyright`)
6. Commit your changes (`git commit -m 'Add some amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Development Guidelines

- Follow [PEP 8](https://pep8.org/) style guidelines (enforced by ruff)
- Write comprehensive tests for new functionality
- Add type hints to all public APIs
- Update documentation for user-facing changes
- Keep the changelog updated

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
