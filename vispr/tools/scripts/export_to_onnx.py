"""Export a PyTorch model to ONNX format for cross-platform deployment."""
import argparse
import torch
import torch.nn as nn

from vispr.tools.common.utils import reload_model_weights
from vispr.models import build_model


def export_to_onnx(weights_path, output_path, arch='resnet50', num_classes=68,
                   input_size=224, model_type='attribute', num_privacy_scores=30,
                   export_privacy_scores=False):
    """Export PyTorch model to ONNX format.

    Args:
        weights_path: Path to PyTorch checkpoint (.pth)
        output_path: Destination .onnx file
        arch: Backbone architecture name (e.g. 'resnet50')
        num_classes: Number of attribute classes (default 68)
        input_size: Image input size (default 224)
        model_type: 'attribute' or 'privacy_aware'
        num_privacy_scores: Output dim of the privacy branch (privacy_aware only)
        export_privacy_scores: If True (and privacy_aware), export a graph
            whose second output is the privacy scores. Otherwise only attribute
            logits are exported.
    """
    model = build_model(arch, num_classes, model_type=model_type,
                        num_privacy_scores=num_privacy_scores)

    # Load weights
    reload_model_weights(model, weights_path, strict=False, map_location='cpu')

    model.eval()

    # Create dummy input (batch_size=1, 3 channels, HxW)
    dummy_input = torch.randn(1, 3, input_size, input_size)

    output_names = ['logits']
    if export_privacy_scores and model_type == 'privacy_aware':
        output_names.append('privacy_scores')

    class _ExportWrapper(nn.Module):
        """Selects the requested outputs from the underlying model."""

        def __init__(self, inner, want_scores):
            super().__init__()
            self.inner = inner
            self.want_scores = want_scores

        def forward(self, x):
            out = self.inner(x)
            if isinstance(out, tuple):
                if self.want_scores:
                    return out
                return out[0]
            return out

    export_model = _ExportWrapper(model,
                                  export_privacy_scores and model_type == 'privacy_aware')
    export_model.eval()

    # Export to ONNX
    torch.onnx.export(
        export_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['images'],
        output_names=output_names,
        verbose=False,
        dynamic_axes={
            'images': {0: 'batch_size'},
            'logits': {0: 'batch_size'}
        }
    )
    print(f'Model exported to ONNX: {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True, help='Path to PyTorch checkpoint')
    parser.add_argument('--output', required=True, help='Output ONNX path')
    parser.add_argument('--arch', default='resnet50')
    parser.add_argument('--num-classes', type=int, default=68)
    parser.add_argument('--input-size', type=int, default=224)
    parser.add_argument('--model-type', default='attribute',
                        choices=['attribute', 'privacy_aware'],
                        help='Model class to use: "attribute" (base) or '
                             '"privacy_aware" (PRCNN extension with privacy score branch)')
    parser.add_argument('--num-privacy-scores', type=int, default=30,
                        help='Number of privacy score outputs for --model-type privacy_aware')
    parser.add_argument('--export-privacy-scores', action='store_true',
                        help='Export a second output containing privacy score predictions')
    args = parser.parse_args()

    export_to_onnx(args.weights, args.output, args.arch, args.num_classes,
                   args.input_size, args.model_type, args.num_privacy_scores,
                   args.export_privacy_scores)