"""Export a PyTorch model to ONNX format for cross-platform deployment."""
import argparse
import torch
import torchvision.models as models


def export_to_onnx(weights_path, output_path, arch='resnet50', num_classes=68, input_size=224):
    """Export PyTorch model to ONNX format."""
    # Load model
    model = getattr(models, arch)(pretrained=False)
    in_f = model.fc.in_features
    model.fc = torch.nn.Linear(in_f, num_classes)

    # Load weights
    state_dict = torch.load(weights_path, map_location='cpu')
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        model.load_state_dict(state_dict['state_dict'])
    else:
        model.load_state_dict(state_dict)

    model.eval()

    # Create dummy input (batch_size=1, 3 channels, HxW)
    dummy_input = torch.randn(1, 3, input_size, input_size)

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['images'],
        output_names=['logits'],
        verbose=False,
        dynamic_axes={
            'images': {0: 'batch_size'},
            'logits': {0: 'batch_size'}
        }
    )
    print(f'✓ Model exported to ONNX: {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True, help='Path to PyTorch checkpoint')
    parser.add_argument('--output', required=True, help='Output ONNX path')
    parser.add_argument('--arch', default='resnet50')
    parser.add_argument('--num-classes', type=int, default=68)
    parser.add_argument('--input-size', type=int, default=224)
    args = parser.parse_args()

    export_to_onnx(args.weights, args.output, args.arch, args.num_classes, args.input_size)

