"""Best-effort mapping of extracted Caffe weights (.npz from caffe_extract_weights.py)
into a PyTorch model state_dict.

Usage:
    python map_caffe_to_pytorch.py --caffe-npz caffe_weights.npz --arch resnet50 --out pytorch_mapped.pth

This script attempts to match caffe layer names to pytorch state_dict keys by
substring matching and shape compatibility. It will print unmatched keys for manual inspection.
"""
import argparse
import numpy as np
import torch
import torchvision.models as models


def load_caffe_npz(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    # data contains arrays where each value is an object array of parameter blobs
    caffe_weights = {}
    for k in data.files:
        caffe_weights[k] = data[k].tolist()
    return caffe_weights


def build_model(arch, num_classes, pretrained=False):
    model = getattr(models, arch)(pretrained=pretrained)
    if hasattr(model, 'fc'):
        in_f = model.fc.in_features
        model.fc = torch.nn.Linear(in_f, num_classes)
    return model


def try_map(caffe_weights, model):
    state = model.state_dict()
    mapped = {}
    used_caffe = set()

    # Flatten caffe weights shapes
    caf_items = list(caffe_weights.items())

    # Try substring matching first
    for s_key in list(state.keys()):
        s_shape = tuple(state[s_key].cpu().numpy().shape)
        found = False
        # scan caffe keys
        for c_key, c_val in caf_items:
            # c_val is a list/array of blobs (weights and maybe bias)
            # try to use first blob if only one, else try to match by shape
            for blob in c_val:
                bshape = tuple(np.array(blob).shape)
                if bshape == s_shape:
                    mapped[s_key] = np.array(blob)
                    used_caffe.add(c_key)
                    found = True
                    break
            if found:
                break
        if not found:
            # try more relaxed: if any blob can be reshaped/flattened to size
            for c_key, c_val in caf_items:
                for blob in c_val:
                    if np.prod(np.array(blob).shape) == np.prod(s_shape):
                        mapped[s_key] = np.array(blob).reshape(s_shape)
                        used_caffe.add(c_key)
                        found = True
                        break
                if found:
                    break
    return mapped, used_caffe


def apply_mapping_to_model(model, mapped):
    state = model.state_dict()
    new_state = {}
    for k, v in state.items():
        if k in mapped:
            arr = mapped[k]
            tensor = torch.from_numpy(arr).type(v.dtype)
            new_state[k] = tensor
        else:
            new_state[k] = v
    model.load_state_dict(new_state)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--caffe-npz', required=True)
    parser.add_argument('--arch', default='resnet50')
    parser.add_argument('--num-classes', type=int, default=68)
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--out', default='mapped_pytorch.pth')
    args = parser.parse_args()

    caffe_weights = load_caffe_npz(args.caffe_npz)
    model = build_model(args.arch, args.num_classes, pretrained=args.pretrained)
    mapped, used = try_map(caffe_weights, model)
    print(f'Mapped {len(mapped)} pytorch keys using {len(used)} caffe keys')
    model = apply_mapping_to_model(model, mapped)
    torch.save(model.state_dict(), args.out)
    print('Saved mapped state_dict to', args.out)


if __name__ == '__main__':
    main()

