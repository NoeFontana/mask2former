import torch
from PIL import Image
from timm.data import config, transforms_factory

from mask2former.modeling.backbone import Backbone


def test_forward_image(sample_image: Image.Image) -> None:
    # Create backbone
    backbone = Backbone()

    transforms = transforms_factory.create_transform(
        **config.resolve_model_data_config(backbone.model), is_training=False
    )

    # Process image and get features
    input_tensor: torch.Tensor = transforms(sample_image).unsqueeze(0)  # type: ignore

    assert input_tensor.shape == (1, 3, 224, 224)

    features = backbone(input_tensor)

    for i, feature in enumerate(features, start=1):
        assert feature.dim() == 4
        assert feature.size(0) == 1  # Batch size
        assert feature.size(1) == 128 * (2**i)  # Channels

        expected_dim = 224 // (2 ** (i + 2))
        assert feature.size(2) == expected_dim
        assert feature.size(3) == expected_dim
