"""PyTorch Dataset for PAP / VISPR-style JSON annotations.

Each annotation file is expected to be a JSON object with at least the
following keys:
- "image_path": path relative to DS_ROOT
- "labels" or "attributes": attribute ids (optional)
- "safe": boolean (optional)

If attribute label vectors are not present, they will be created using
the utilities in `vispr.tools.common.utils`.

Optionally loads per-image user privacy preference scores from a TSV
file (``user_scores_path``) for privacy-aware training.  The TSV is
expected to have a header row with columns ``image_id`` (or
``image_path``) followed by ``score_0`` .. ``score_29`` (or any 30
numeric columns).  When provided, ``__getitem__`` returns a 3-tuple
``(image_tensor, label_vec, user_scores)`` instead of the usual 2-tuple.
"""
from typing import List, Optional, Tuple
import csv
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
                 transform: Optional[SimpleTransformer] = None, ds_root: Optional[str] = None,
                 user_scores_path: Optional[str] = None):
        self.ds_root = ds_root if ds_root is not None else DS_ROOT
        self.im_shape = tuple(im_shape)
        self.transform = transform if transform is not None else SimpleTransformer(mean=[104, 117, 123])
        self.with_user_scores = user_scores_path is not None

        # Load annotations (list of paths, one per line)
        with open(anno_list_path, 'r') as f:
            lines = [l.strip() for l in f if l.strip() != '']

        # Each line typically points to a JSON annotation file
        self.anno_paths = [osp.join(self.ds_root, l) if not osp.isabs(l) else l for l in lines]
        self.annos = [json.load(open(p)) for p in self.anno_paths]

        # Load attribute mapping
        self.attr_id_to_name, self.attr_id_to_idx = load_attributes()

        # Load per-image user privacy preference scores (optional)
        self.user_scores = self._load_user_scores(user_scores_path)

    def __len__(self):
        return len(self.annos)

    def _load_user_scores(self, user_scores_path: Optional[str]) -> dict:
        """Load user privacy preference scores from a TSV file.

        Expected format (with optional header):
            <image_id or image_path>  <score_0>  <score_1> ... <score_N-1>

        Returns a dict mapping the first column (e.g. the image filename
        or annotation path) to a ``np.ndarray`` of float scores.  Returns
        an empty dict when ``user_scores_path`` is None.
        """
        if user_scores_path is None:
            return {}

        scores = {}
        with open(user_scores_path, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            rows = list(reader)
        if not rows:
            return scores

        # Treat first row as data if its final cell parses as float,
        # otherwise treat it as a header row.
        first_row = rows[0]
        has_header = False
        if len(first_row) >= 2:
            try:
                float(first_row[-1])
            except (ValueError, IndexError):
                has_header = True
        else:
            has_header = True

        data_rows = rows[1:] if has_header else rows
        for row in data_rows:
            if len(row) < 2:
                continue
            key = row[0].strip()
            try:
                vec = np.array([float(v) for v in row[1:]], dtype=np.float32)
            except ValueError:
                continue
            if vec.size > 0:
                scores[key] = vec
        return scores

    def _lookup_user_scores(self, idx: int):
        """Return the user-scores vector for the sample at ``idx``.

        Falls back to a zero vector of the same width as the first loaded
        row when no match is found.
        """
        anno = self.annos[idx]
        # Fast path: exact key match first
        for key in (osp.basename(str(anno.get('image_path', ''))),
                    str(anno.get('image_path', '')),
                    str(anno.get('anno_path', ''))):
            if key and key in self.user_scores:
                return torch.from_numpy(self.user_scores[key])
        if self.user_scores:
            width = next(iter(self.user_scores.values())).shape[0]
            return torch.zeros(width, dtype=torch.float32)
        return torch.zeros(0, dtype=torch.float32)

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

        if self.with_user_scores:
            user_scores = self._lookup_user_scores(idx)
            return tensor, torch.from_numpy(label_vec), user_scores

        return tensor, torch.from_numpy(label_vec)

