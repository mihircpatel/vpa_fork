"""Privacy-aware attribute prediction model (PRCNN extension).

Extends :class:`AttributeModel` with a privacy scoring branch that predicts
30-dimensional user privacy preference scores from the 68-dimensional
attribute logits.

Architecture (mirrors ``googlenet-prcnn/train_val.prototxt`` lines 2121-2220)::

    attribute logits (B, 68)
        │
        ├─ fc_ps_1: Linear(68 → 128) + Sigmoid
        ├─ fc_ps_2: Linear(128 → 128) + Sigmoid
        └─ fc9:     Linear(128 → num_privacy_scores)

    Returns (attr_logits, privacy_scores)

The base backbone and attribute classifier weights are fully compatible with
checkpoints saved for :class:`AttributeModel`.  Use ``strict=False`` when
loading a base checkpoint into this model; the privacy branch will be
randomly initialised.
"""
import torch
import torch.nn as nn

from vispr.models.attribute_model import AttributeModel


class PrivacyAwareAttributeModel(AttributeModel):
    """Attribute model extended with a privacy scoring branch.

    Inherits the backbone and68-dim attribute classifier from
    :class:`AttributeModel`.  Adds three linear layers (with Sigmoid
    activations) that map the 68 attribute logits to a
    ``num_privacy_scores``-dimensional privacy score vector.

    Forward pass returns a **tuple** ``(attr_logits, privacy_scores)``.
    For attribute-only inference, use ``attr_logits = output[0]``.

    Args:
        arch: torchvision backbone name.
        num_classes: number of attribute classes (default 68).
        num_privacy_scores: number of privacy score outputs (default 30,
            matching the 30 user-profile dimensions in the original Caffe
            prototxt).
        pretrained: if True, load ImageNet-pretrained backbone weights.

    Example::

        model = PrivacyAwareAttributeModel('resnet50', 68, 30)
        attr_logits, priv_scores = model(images)

        # Load base checkpoint into backbone only
        base_ckpt = torch.load('base_model.pth')
        model.load_state_dict(base_ckpt['state_dict'], strict=False)
    """

    def __init__(self, arch: str = 'resnet50', num_classes: int = 68,
                 num_privacy_scores: int = 30, pretrained: bool = False):
        super().__init__(arch, num_classes, pretrained)
        self.num_privacy_scores = num_privacy_scores

        # Privacy scoring branch
        # Matches prototxt: fc_ps_1 → sigmoid → fc_ps_2 → sigmoid → fc9
        self.privacy_branch = nn.Sequential(
            nn.Linear(num_classes, 128),   # fc_ps_1
            nn.Sigmoid(),                   # relu_ps_1 (named relu but is Sigmoid)
            nn.Linear(128, 128),            # fc_ps_2
            nn.Sigmoid(),                   # relu_ps_2
            nn.Linear(128, num_privacy_scores),  # fc9
        )

        # Xavier-init the privacy branch (matches prototxt weight_filler)
        self._init_privacy_branch()

    def _init_privacy_branch(self):
        """Xavier-initialise the privacy branch linear layers."""
        for m in self.privacy_branch.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        """Forward pass through backbone + attribute classifier + privacy branch.

        Args:
            x: input image tensor of shape ``(B, 3, H, W)``.

        Returns:
            Tuple of:
            - **attr_logits** — attribute logits, shape ``(B, num_classes)``
            - **privacy_scores** — privacy score predictions, shape
              ``(B, num_privacy_scores)``
        """
        attr_logits = super().forward(x)  # (B, num_classes)
        privacy_scores = self.privacy_branch(attr_logits)  # (B, num_privacy_scores)
        return attr_logits, privacy_scores
