import timm
import torch


class Backbone(torch.nn.Module):
    """
    A thin timm backbone wrapper for feature extraction.
    """

    def __init__(
        self,
        model_name: str = "convnext_base.dinov3_lvd1689m",
        out_indices: tuple[int, ...] = (1, 2, 3),
        exportable: bool = True,
    ) -> None:
        """Initialize the backbone model.

        Args:
            model_name: Name of the timm model to use for feature extraction.
            out_indices: Which feature map indices to return from the backbone.
                Typically, (1, 2, 3) for convnext and (2, 3, 4) for resnet.
            exportable: Whether the model should be exportable.
        """
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=True,
            features_only=True,
            exportable=exportable,
            out_indices=out_indices,
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Forward pass through the backbone.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            List of feature tensors at different scales
        """
        return self.model(x)
