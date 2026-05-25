"""Extract weights from a Caffe model and save to a .npz file.

This is a best-effort utility that requires a working Caffe Python
installation. It will iterate over all layers in the Caffe net and save
their parameters (if any) into a numpy archive keyed by the layer name.
The resulting .npz can be used as a starting point when mapping parameters
to a PyTorch model.
"""
import argparse
import os
import numpy as np

try:
    import caffe
except Exception as e:
    raise RuntimeError('Caffe Python module not found. Set VISPR_CAFFE_ROOT or install pycaffe.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('deploy', help='Path to deploy prototxt')
    parser.add_argument('caffemodel', help='Path to caffemodel')
    parser.add_argument('--out', help='Output .npz path', default='caffe_weights.npz')
    args = parser.parse_args()

    net = caffe.Net(args.deploy, args.caffemodel, caffe.TEST)
    weights = {}
    for name, params in net.params.items():
        # params is a list of blobs (weights, bias etc.)
        arrays = [p.data.copy() for p in params]
        weights[name] = arrays
    # Save all weights
    # Note: np.savez cannot save lists of ndarrays directly in a friendly way; we store as object array
    np.savez(args.out, **{k: np.array(v, dtype=object) for k, v in weights.items()})
    print('Saved caffe weights to', args.out)


if __name__ == '__main__':
    main()

