"""Model definitions for VPA attribute prediction.

Provides:
- AttributeModel: Base model (backbone + 68-dim attribute classifier)
- PrivacyAwareAttributeModel: Extended model adding privacy score prediction
  (68-dim attributes + 30-dim privacy scores)
"""
from vispr.models.attribute_model import AttributeModel
from vispr.models.privacy_aware_model import PrivacyAwareAttributeModel


def build_model(arch: str = 'resnet50', num_classes: int = 68,
                model_type: str = 'attribute',
                num_privacy_scores: int = 30,
                pretrained: bool = False):
    """Factory function to build a model by type.

    Args:
        arch: torchvision backbone name (e.g. 'resnet50', 'resnet18')
        num_classes: number of attribute classes (default 68)
        model_type: 'attribute' for base model, 'privacy_aware' for PRCNN extension
        num_privacy_scores: number of privacy score outputs (default 30, for privacy_aware only)
        pretrained: whether to load pretrained backbone weights

    Returns:
        nn.Module: either AttributeModel or PrivacyAwareAttributeModel
    """
    if model_type == 'privacy_aware':
        return PrivacyAwareAttributeModel(
            arch=arch,
            num_classes=num_classes,
            num_privacy_scores=num_privacy_scores,
            pretrained=pretrained,
        )
    return AttributeModel(
        arch=arch,
        num_classes=num_classes,
        pretrained=pretrained,
    )
