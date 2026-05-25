"""A lightweight image transformer compatible with a minimal subset of
the original Caffe SimpleTransformer behavior.

This transformer operates on PIL Images or numpy arrays and returns a
float32 numpy array in shape (C, H, W) with mean subtraction and channel
order conversion (RGB->BGR) to match common Caffe preprocessing.
"""
from typing import Sequence, Tuple, Optional
import numpy as np
from PIL import Image


class SimpleTransformer:
    """Minimal transformer implementing preprocess(image) used in the repo.

    Options supported:
    - mean: iterable of length 3 (BGR mean values) subtracted from channels
    - channel_swap: tuple to reorder channels (e.g. (2,1,0) for RGB->BGR)
    - raw_scale: multiply input by this value (useful if image is in [0,1])
    - transpose: if True will transpose HWC -> CHW
    """

    def __init__(self, mean: Optional[Sequence[float]] = None,
                 channel_swap: Tuple[int, int, int] = (2, 1, 0),
                 raw_scale: float = 255.0,
                 transpose: bool = True):
        self.mean = np.array(mean, dtype=np.float32) if mean is not None else None
        self.channel_swap = channel_swap
        self.raw_scale = raw_scale
        self.transpose = transpose

    def _to_numpy(self, img):
        if isinstance(img, Image.Image):
            arr = np.asarray(img)
        else:
            arr = np.array(img)
        # Ensure RGB
        if arr.ndim == 2:
            # grayscale -> RGB
            arr = np.stack([arr, arr, arr], axis=2)
        if arr.shape[2] == 4:
            # RGBA -> RGB
            arr = np.asarray(Image.fromarray(arr).convert('RGB'))
        return arr.astype(np.float32)

    def preprocess(self, img):
        """Preprocess image and return numpy array shape (C,H,W) float32.

        Expects img as PIL Image or numpy array in HxWxC (RGB) with values
        in [0,255] or [0,1]. If values appear in [0,1], raw_scale is applied.
        """
        arr = self._to_numpy(img)
        # If values appear to be in [0,1], scale up
        if arr.max() <= 1.0:
            arr = arr * self.raw_scale

        # Channel swap (default: RGB->BGR)
        if self.channel_swap is not None:
            arr = arr[:, :, list(self.channel_swap)]

        # Subtract mean (expects [B, G, R])
        if self.mean is not None:
            # Broadcast mean of shape (3,) over H and W
            arr -= self.mean[np.newaxis, np.newaxis, :]

        if self.transpose:
            # HWC -> CHW
            arr = arr.transpose((2, 0, 1))

        return arr.astype(np.float32)

