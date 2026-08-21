"""PyTorch Dataset adapter for streaming tar archives from Hugging Face Hub.

Provides a drop-in replacement for PAPDataset that streams data instead of
loading from local disk.
"""
import random
import logging
from typing import Tuple, Optional, List, Dict, Any
from collections import deque

import torch
import numpy as np
from torch.utils.data import IterableDataset
from PIL import Image

from vispr.tools.common.utils import load_attributes, labels_to_vec
from vispr.torch_utils.transformer import SimpleTransformer

from .hf_tar_streamer import HFTarStreamer
from .dual_archive_streamer import HFDualArchiveStreamer
from .config import StreamingConfig

logger = logging.getLogger(__name__)


class StreamingPAPDataset(IterableDataset):
    """Streaming PyTorch Dataset that loads data from HF tar archives.

    This dataset streams .tar.gz archives from Hugging Face Hub without
    downloading to disk, making it suitable for environments with limited
    disk space (e.g., Google Colab).

    It maintains an in-memory shuffle buffer for randomization during training.
    """

    def __init__(
        self,
        config: StreamingConfig,
        im_shape: Optional[Tuple[int, int]] = None,
        transform: Optional[SimpleTransformer] = None,
        shuffle: bool = True,
        attr_list_path: Optional[str] = None,
        return_metadata: bool = False
    ):
        """Initialize streaming dataset.

        Args:
            config: StreamingConfig with HF repo and file information
            im_shape: Image shape as (height, width). Defaults to config.im_shape
            transform: Image transformer. If None, uses SimpleTransformer with config.mean
            shuffle: Whether to shuffle data using buffer
            attr_list_path: Path to attributes.tsv (optional, uses default if None)
        """
        super().__init__()

        self.config = config
        self.config.validate()

        self.im_shape = tuple(im_shape) if im_shape else config.im_shape
        self.shuffle = shuffle
        self.buffer_size = config.buffer_size
        self.return_metadata = return_metadata

        # Initialize transformer
        if transform is not None:
            self.transform = transform
        else:
            self.transform = SimpleTransformer(mean=list(config.mean))

        # Load attribute mappings
        self.attr_id_to_name, self.attr_id_to_idx = load_attributes(attr_list_path)

        # Initialize streamer based on mode (dual archive vs combined archive)
        if config.is_dual_archive_mode():
            # Dual archive mode: separate image and annotation archives
            self.streamer = HFDualArchiveStreamer(
                repo_id=config.repo_id,
                image_archive_path=config.image_archive_path,
                annotation_archive_path=config.annotation_archive_path,
                anno_list_path=config.anno_list_path
            )
            logger.info(
                f"Initialized StreamingPAPDataset (dual archive mode): "
                f"repo={config.repo_id}, images={config.image_archive_path}, "
                f"annotations={config.annotation_archive_path}, "
                f"anno_list={config.anno_list_path}, buffer_size={self.buffer_size}, "
                f"shuffle={shuffle}"
            )
        else:
            # Combined archive mode: images and annotations in single archive
            self.streamer = HFTarStreamer(
                repo_id=config.repo_id,
                file_path=config.file_path
            )
            logger.info(
                f"Initialized StreamingPAPDataset (combined archive mode): "
                f"repo={config.repo_id}, file={config.file_path}, "
                f"buffer_size={self.buffer_size}, shuffle={shuffle}"
            )

    def _process_record(self, record: Dict[str, Any]):
        """Process a single record into model input format.

        Args:
            record: Record dictionary with 'image' and 'labels'

        Returns:
            Tuple of (image_tensor, label_tensor) or (image_tensor, label_tensor, image_path)
            when return_metadata=True.
        """
        # Get image
        img = record['image']

        # Resize image
        img = img.resize((self.im_shape[1], self.im_shape[0]), Image.LANCZOS)

        # Build label vector
        if 'label_vec' in record:
            label_vec = np.array(record['label_vec'], dtype=np.float32)
        else:
            labels = set(record.get('labels', []))
            label_vec = labels_to_vec(labels, self.attr_id_to_idx).astype(np.float32)

        # Apply transformation
        data = self.transform.preprocess(img)

        # Convert to torch tensors
        image_tensor = torch.from_numpy(data.copy())
        label_tensor = torch.from_numpy(label_vec)
        image_path = record.get('image_path', record.get('anno_path'))

        if self.return_metadata:
            return image_tensor, label_tensor, image_path
        return image_tensor, label_tensor

    def _shuffle_buffer(self, iterator):
        """Apply shuffle buffer to iterator.

        Maintains a buffer of samples and yields random samples from it.
        This provides approximate shuffling without loading entire dataset.

        Args:
            iterator: Iterator yielding records

        Yields:
            Shuffled records
        """
        buffer = deque(maxlen=self.buffer_size)

        # Fill buffer initially
        for i, item in enumerate(iterator):
            buffer.append(item)
            if i >= self.buffer_size - 1:
                break

        # Yield random items and refill
        for item in iterator:
            if len(buffer) > 0:
                idx = random.randint(0, len(buffer) - 1)
                yield buffer[idx]
                buffer[idx] = item

        # Yield remaining items in buffer randomly
        while len(buffer) > 0:
            idx = random.randint(0, len(buffer) - 1)
            yield buffer[idx]
            del buffer[idx]

    def __iter__(self):
        """Iterate over dataset samples.

        Yields:
            Tuple of (image_tensor, label_tensor)
        """
        # Get record iterator from streamer
        record_iterator = self.streamer.extract_structured_data()

        # Apply shuffle buffer if needed
        if self.shuffle:
            record_iterator = self._shuffle_buffer(record_iterator)

        # Process and yield records
        for record in record_iterator:
            try:
                yield self._process_record(record)
            except Exception as e:
                logger.warning(f"Failed to process record {record.get('image_path', 'unknown')}: {e}")
                continue

    def __len__(self):
        """Get dataset length.

        Note: For streaming datasets, exact length is unknown without
        reading entire archive. Returns approximate count if available,
        otherwise returns -1.
        """
        # Try to get estimate (expensive, requires full scan)
        # Only do this if explicitly needed
        return -1


class StreamingPAPDatasetFromFile(StreamingPAPDataset):
    """Convenience wrapper that creates StreamingConfig from parameters.

    This provides an interface similar to the original PAPDataset for easier
    integration into existing code.
    """

    def __init__(
        self,
        repo_id: str,
        file_path: str,
        im_shape: Tuple[int, int] = (224, 224),
        transform: Optional[SimpleTransformer] = None,
        buffer_size: int = 1000,
        shuffle: bool = True,
        attr_list_path: Optional[str] = None,
        mean: Tuple[float, float, float] = (104.0, 117.0, 123.0)
    ):
        """Initialize streaming dataset with explicit parameters.

        Args:
            repo_id: Hugging Face repository ID
            file_path: Path to .tar.gz file in repository
            im_shape: Image shape (height, width)
            transform: Optional image transformer
            buffer_size: Shuffle buffer size
            shuffle: Whether to shuffle data
            attr_list_path: Path to attributes.tsv
            mean: Mean values for normalization [B, G, R]
        """
        config = StreamingConfig(
            data_source='hf_tar_stream',
            repo_id=repo_id,
            file_path=file_path,
            buffer_size=buffer_size,
            im_shape=im_shape,
            mean=mean
        )

        super().__init__(
            config=config,
            im_shape=im_shape,
            transform=transform,
            shuffle=shuffle,
            attr_list_path=attr_list_path
        )


def get_streaming_dataloader(config: StreamingConfig, shuffle: bool = True, **loader_kwargs):
    """Create a DataLoader for streaming dataset.

    Args:
        config: StreamingConfig instance
        shuffle: Whether to shuffle data
        **loader_kwargs: Additional arguments for DataLoader (batch_size, num_workers, etc.)

    Returns:
        torch.utils.data.DataLoader instance
    """
    from torch.utils.data import DataLoader

    dataset = StreamingPAPDataset(config=config, shuffle=shuffle)

    # Default loader kwargs
    default_kwargs = {
        'batch_size': config.batch_size,
        'num_workers': 0,  # IterableDataset works best with num_workers=0
    }
    default_kwargs.update(loader_kwargs)

    return DataLoader(dataset, **default_kwargs)
