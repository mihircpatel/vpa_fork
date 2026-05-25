"""Run a small set of unit-style smoke tests for the PyTorch pipeline.

This creates a temporary image and annotation and verifies that
PAPDataset and a small model can run a forward pass.
"""
import os
import json
import tempfile
from PIL import Image
import numpy as np
import torch
import torchvision.models as models
from vispr.datasets.pap_dataset import PAPDataset


def create_dummy_image(path, size=(224, 224)):
    img = Image.fromarray((np.random.rand(size[0], size[1], 3) * 255).astype('uint8'))
    img.save(path)


def main():
    tmp = tempfile.mkdtemp()
    img_path = os.path.join(tmp, 'img1.jpg')
    create_dummy_image(img_path)

    # Create annotation JSON
    anno = {'image_path': img_path, 'labels': [] , 'safe': True}
    anno_path = os.path.join(tmp, 'anno1.json')
    with open(anno_path, 'w') as f:
        json.dump(anno, f)

    # Create list file
    list_path = os.path.join(tmp, 'list.txt')
    with open(list_path, 'w') as f:
        f.write(anno_path + '\n')

    dataset = PAPDataset(list_path, im_shape=(224, 224))
    x, y = dataset[0]
    print('Dataset sample shapes:', x.shape, y.shape)

    # Run through a tiny model
    model = models.resnet18(pretrained=False)
    in_f = model.fc.in_features
    model.fc = torch.nn.Linear(in_f, len(y))
    model.eval()
    with torch.no_grad():
        inp = torch.from_numpy(x).unsqueeze(0).float()
        out = model(inp)
        print('Model forward OK, output shape:', out.shape)

    print('All smoke tests passed')


if __name__ == '__main__':
    main()

