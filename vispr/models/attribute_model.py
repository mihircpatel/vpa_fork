"""Base attribute prediction model.

Wraps a torchvision backbone with a linear classifier head to produce
68-dimensional attribute logits. State-dict keys are identical to the
raw torchvision model (backbone layers + ``fc``), so existing ResNet
checkpoints load without modification.
"""
import torch
import torch.nn as nn
import torchvision.models as models


class AttributeModel(nn.Module):
    """Base model: torchvision backbone + linear attribute classifier.

    This class formalises the inline ``build_model()`` pattern that already
    exists in the training and inference scripts.  The backbone and final
    linear layer names match torchvision conventions so that old checkpoints
    are fully compatible.

    Args:
        arch: torchvision backbone name (e.g. ``'resnet50'``, ``'resnet18'``).
        num_classes: number of output attribute classes (default 68).
        pretrained: if True, load ImageNet-pretrained backbone weights.
    """

    def __init__(self, arch: str = 'resnet50', num_classes: int = 68,
                 pretrained: bool = False):
        super().__init__()
        arch = arch.lower()
        self.arch = arch
        self.num_classes = num_classes

        if hasattr(models, arch):
            self.backbone = getattr(models, arch)(pretrained=pretrained)
        else:
            raise ValueError(f'Unsupported architecture: {arch}')

        # Replace the final fully-connected layer to output num_classes logits.
        # For ResNet-family models the head is ``self.backbone.fc``.
        # For MobileNet-family models the head is the last Linear in
        # ``self.backbone.classifier``.
        if hasattr(self.backbone, 'fc'):
            in_f = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_f, num_classes)
        elif hasattr(self.backbone, 'classifier'):
            # MobileNetV2/V3 classifier is Sequential(..., Linear(...))
            in_f = self.backbone.classifier[-1].in_features
            self.backbone.classifier[-1] = nn.Linear(in_f, num_classes)
        else:
            raise ValueError(
                f'Cannot locate classifier head on {arch}. '
                'Expected model.fc or model.classifier.'
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through backbone + classifier.

        Args:
            x: input image tensor of shape ``(B, 3, H, W)``.

        Returns:
            Attribute logits of shape ``(B, num_classes)``.
        """
        return self.backbone(x)
