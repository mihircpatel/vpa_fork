"""Run attribute predictions using a PyTorch model.

This script is a lightweight replacement for the original Caffe-based
`attribute_predict.py`. It uses a PyTorch model (e.g., torchvision backbone
with a final linear layer) to compute per-attribute probabilities.

Example usage:
    python attribute_predict_torch.py --arch resnet50 --weights my_model.pth \
        --infile test2017.txt --outfile preds.jsonl --batch-size 32 --num-classes 68
"""
import argparse
import os
import os.path as osp
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models

from vispr.datasets.pap_dataset import PAPDataset


def build_model(arch: str, num_classes: int, pretrained: bool = False):
    arch = arch.lower()
    if arch.startswith('resnet'):
        model = getattr(models, arch)(pretrained=pretrained)
        # Replace final fc
        in_f = model.fc.in_features
        model.fc = nn.Linear(in_f, num_classes)
        return model
    elif arch.startswith('mobilenet'):
        model = getattr(models, arch)(pretrained=pretrained)
        in_f = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_f, num_classes)
        return model
    else:
        raise ValueError('Unsupported arch: {}'.format(arch))


def classify_paths(model, device, dataset: PAPDataset, out_file: str, batch_size: int = 64):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    model = model.to(device)
    model.eval()

    with open(out_file, 'w') as wf:
        idx = 0
        for images, _ in loader:
            images = images.to(device).float()
            with torch.no_grad():
                outputs = model(images)
                probs = torch.sigmoid(outputs).cpu().numpy()
            # For each item in batch, write JSON line
            batch_size_local = probs.shape[0]
            for b in range(batch_size_local):
                ann_path = dataset.anno_paths[idx]
                entry = {'anno_path': ann_path, 'pred_probs': probs[b].tolist()}
                wf.write(json.dumps(entry) + '\n')
                idx += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='resnet50')
    parser.add_argument('--weights', type=str, default=None, help='Path to model weights (.pth)')
    parser.add_argument('--infile', type=str, required=True, help='List of annotation paths to classify')
    parser.add_argument('--outfile', type=str, required=True, help='Output JSONL file for predictions')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-classes', type=int, default=68)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained backbone')
    args = parser.parse_args()

    model = build_model(args.arch, args.num_classes, pretrained=args.pretrained)
    if args.weights is not None:
        ckpt = torch.load(args.weights, map_location='cpu')
        # Support either state_dict or full model checkpoint
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            state = ckpt['state_dict']
        else:
            state = ckpt
        # Some checkpoints have 'module.' prefixes from DataParallel
        new_state = {}
        for k, v in state.items():
            nk = k.replace('module.', '')
            new_state[nk] = v
        model.load_state_dict(new_state, strict=False)

    device = torch.device(args.device)
    dataset = PAPDataset(args.infile, im_shape=(224, 224))
    classify_paths(model, device, dataset, args.outfile, batch_size=args.batch_size)


if __name__ == '__main__':
    main()

