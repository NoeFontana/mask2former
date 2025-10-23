"""Test configuration and fixtures."""

from urllib.request import urlopen

import pytest
from PIL import Image


@pytest.fixture
def sample_image():
    """Sample image for testing."""
    return Image.open(
        urlopen(
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png"
        )
    )
