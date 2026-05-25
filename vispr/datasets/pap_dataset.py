"""PyTorch Dataset for PAP / VISPR-style JSON annotations.

Each annotation file is expected to be a JSON object with at least the
following keys:
- "image_path": path relative to DS_ROOT
- "labels" or "attributes": attribute ids (optional)
- "safe": boolean (optional)

If attribute label vectors are not present, they will be created using
the utilities in `vispr.tools.common.utils`.
"""
from typing import List, Optional, Tuple
import os
import os.path as osp
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch

from vispr import DS_ROOT
from vispr.tools.common.utils import load_attributes, labels_to_vec
from vispr.torch_utils.transformer import SimpleTransformer


class PAPDataset(Dataset):
    def __init__(self, anno_list_path: str, im_shape: Tuple[int, int] = (227, 227),
                 transform: Optional[SimpleTransformer] = None, ds_root: Optional[str] = None):
        self.ds_root = ds_root if ds_root is not None else DS_ROOT
        self.im_shape = tuple(im_shape)
        self.transform = transform if transform is not None else SimpleTransformer(mean=[104, 117, 123])

        # Load annotations (list of paths, one per line)
        with open(anno_list_path, 'r') as f:
            lines = [l.strip() for l in f if l.strip() != '']

        # Each line typically points to a JSON annotation file
        self.anno_paths = [osp.join(self.ds_root, l) if not osp.isabs(l) else l for l in lines]
        self.annos = [json.load(open(p)) for p in self.anno_paths]

        # Load attribute mapping
        self.attr_id_to_name, self.attr_id_to_idx = load_attributes()

    def __len__(self):
        return len(self.annos)

    def _load_image(self, image_path: str) -> Image.Image:
        if not osp.isabs(image_path):
            image_path = osp.join(self.ds_root, image_path)
        img = Image.open(image_path).convert('RGB')
        # Resize maintaining aspect ratio and then center crop to im_shape if needed
        img = img.resize((self.im_shape[1], self.im_shape[0]), Image.LANCZOS)
        return img

    def __getitem__(self, idx: int):
        anno = self.annos[idx]
        image_path = anno.get('image_path')
        img = self._load_image(image_path)

        # Build label vector
        if 'label_vec' in anno:
            label_vec = np.array(anno['label_vec'], dtype=np.float32)
        else:
            # If 'labels' present use that, otherwise use 'attributes' dict
            if 'labels' in anno:
                labels = set(anno['labels'])
            else:
                labels = set()
                if 'attributes' in anno:
                    for _, attr_list in anno['attributes'].items():
                        labels.update(attr_list)
            label_vec = labels_to_vec(labels, self.attr_id_to_idx).astype(np.float32)

        data = self.transform.preprocess(img)

        # Convert to torch tensor
        tensor = torch.from_numpy(data.copy())

        return tensor, torch.from_numpy(label_vec)

