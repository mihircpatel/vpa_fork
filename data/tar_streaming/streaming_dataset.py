"""PyTorch Dataset adapter for streaming tar archives.

Provides a drop-in replacement for PAPDataset that streams data from tar.gz
archives instead of loading from individual local files. Supports streaming
from both Hugging Face Hub (remote) and local disk, with optional
record-level caching to avoid repeated streaming on subsequent runs.
"""
import os
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
from .local_tar_streamer import LocalTarStreamer
from .config import StreamingConfig

logger = logging.getLogger(__name__)


class StreamingPAPDataset(IterableDataset):
    """Streaming PyTorch Dataset that loads data from tar archives.

    This dataset streams .tar.gz archives without writing them to a directory
    tree on disk, making it suitable for environments with limited disk space.
    It supports reading from Hugging Face Hub (`data_source='hf_tar_stream'`)
    or from local disk (`data_source='local_tar_stream'`).

    It maintains an in-memory shuffle buffer for randomization during training.
    Optionally caches processed records locally to avoid re-streaming.
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
            config: StreamingConfig with archive and source information
            im_shape: Image shape as (height, width). Defaults to config.im_shape
            transform: Image transformer. If None, uses SimpleTransformer with config.mean
            shuffle: Whether to shuffle data using buffer
            attr_list_path: Path to attributes.tsv (optional, uses default if None)
            return_metadata: If True, yield (image, label, image_path) tuples
        """
        super().__init__()

        self.config = config
        self.config.validate()

        self.im_shape = tuple(im_shape) if im_shape else config.im_shape
        self.shuffle = shuffle
        self.buffer_size = config.buffer_size
        self.return_metadata = return_metadata
        self.log_interval = config.log_interval

        # Initialize transformer
        if transform is not None:
            self.transform = transform
        else:
            self.transform = SimpleTransformer(mean=list(config.mean))

        # Load attribute mappings
        self.attr_id_to_name, self.attr_id_to_idx = load_attributes(attr_list_path)

        # Cache configuration
        self._cache_path = config.get_cache_path()
        self._cache_enabled = self._cache_path is not None

        # Initialize streamer based on mode (dual vs combined) and source (local vs remote)
        is_local = config.is_local_streaming_mode()
        if is_local:
            if config.is_dual_archive_mode():
                self.streamer = LocalTarStreamer(
                    image_archive_path=config.image_archive_path,
                    annotation_archive_path=config.annotation_archive_path,
                    anno_list_path=config.anno_list_path,
                )
                logger.info(
                    f"Initialized StreamingPAPDataset (local dual archive mode): "
                    f"images={config.image_archive_path}, "
                    f"annotations={config.annotation_archive_path}, "
                    f"anno_list={config.anno_list_path}, buffer_size={self.buffer_size}, "
                    f"shuffle={shuffle}, cache={'on' if self._cache_enabled else 'off'}"
                )
            else:
                self.streamer = LocalTarStreamer(
                    file_path=config.file_path,
                )
                logger.info(
                    f"Initialized StreamingPAPDataset (local combined archive mode): "
                    f"file={config.file_path}, "
                    f"buffer_size={self.buffer_size}, shuffle={shuffle}, "
                    f"cache={'on' if self._cache_enabled else 'off'}"
                )
        elif config.is_dual_archive_mode():
            self.streamer = HFDualArchiveStreamer(
                repo_id=config.repo_id,
                image_archive_path=config.image_archive_path,
                annotation_archive_path=config.annotation_archive_path,
                anno_list_path=config.anno_list_path,
                max_retries=config.max_retries,
            )
            logger.info(
                f"Initialized StreamingPAPDataset (HF dual archive mode): "
                f"repo={config.repo_id}, images={config.image_archive_path}, "
                f"annotations={config.annotation_archive_path}, "
                f"anno_list={config.anno_list_path}, buffer_size={self.buffer_size}, "
                f"shuffle={shuffle}, cache={'on' if self._cache_enabled else 'off'}"
            )
        else:
            self.streamer = HFTarStreamer(
                repo_id=config.repo_id,
                file_path=config.file_path,
                max_retries=config.max_retries,
            )
            logger.info(
                f"Initialized StreamingPAPDataset (HF combined archive mode): "
                f"repo={config.repo_id}, file={config.file_path}, "
                f"buffer_size={self.buffer_size}, shuffle={shuffle}, "
                f"cache={'on' if self._cache_enabled else 'off'}"
            )

        # Internal counters for progress logging
        self._yielded_count = 0
        self._error_count = 0

    def _process_record(self, record: Dict[str, Any]):
        """Process a single record into model input format.

        Args:
            record: Record dictionary with 'image' and 'labels'

        Returns:
            Tuple of (image_tensor, label_tensor) or (image_tensor, label_tensor, image_path)
            when return_metadata=True.
        """
        img = record['image']
        img = img.resize((self.im_shape[1], self.im_shape[0]), Image.LANCZOS)

        if 'label_vec' in record:
            label_vec = np.array(record['label_vec'], dtype=np.float32)
        else:
            labels = set(record.get('labels', []))
            label_vec = labels_to_vec(labels, self.attr_id_to_idx).astype(np.float32)

        data = self.transform.preprocess(img)

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

        for i, item in enumerate(iterator):
            buffer.append(item)
            if i >= self.buffer_size - 1:
                break

        for item in iterator:
            if len(buffer) > 0:
                idx = random.randint(0, len(buffer) - 1)
                yield buffer[idx]
                buffer[idx] = item

        while len(buffer) > 0:
            idx = random.randint(0, len(buffer) - 1)
            yield buffer[idx]
            del buffer[idx]

    def _iter_from_cache(self):
        """Iterate over cached records.

        Yields:
            Tuples of (image_tensor, label_tensor) or (image_tensor, label_tensor, image_path)
        """
        cache_dir = self._cache_path
        idx = 0
        while True:
            path = os.path.join(cache_dir, f"{idx}.pt")
            if not os.path.exists(path):
                break
            try:
                data = torch.load(path, weights_only=False)
                image_tensor = data['image']
                label_tensor = data['label']
                if self.return_metadata:
                    image_path = data.get('image_path', None)
                    yield image_tensor, label_tensor, image_path
                else:
                    yield image_tensor, label_tensor
                idx += 1
            except Exception as e:
                logger.warning(f"Failed to load cached record {path}: {e}")
                idx += 1
                continue

        logger.info(f"Loaded {idx} records from cache: {cache_dir}")

    def _write_cache_record(self, record_tuple, idx: int):
        """Write a single processed record to cache.

        Args:
            record_tuple: The (image_tensor, label_tensor) or (image_tensor, label_tensor, image_path) tuple
            idx: Record index for filename
        """
        cache_dir = self._cache_path
        os.makedirs(cache_dir, exist_ok=True)

        path = os.path.join(cache_dir, f"{idx}.pt")
        data = {
            'image': record_tuple[0],
            'label': record_tuple[1],
        }
        if self.return_metadata and len(record_tuple) > 2:
            data['image_path'] = record_tuple[2]

        try:
            torch.save(data, path)
        except Exception as e:
            logger.warning(f"Failed to write cache record {path}: {e}")

    def __iter__(self):
        """Iterate over dataset samples.

        If cache is enabled and populated, reads from cache.
        Otherwise streams from the configured source (HF Hub or local archive)
        and optionally populates cache.

        Yields:
            Tuple of (image_tensor, label_tensor) or (image_tensor, label_tensor, image_path)
        """
        self._yielded_count = 0
        self._error_count = 0

        # Check cache first
        if self._cache_enabled and self._is_cache_populated():
            logger.info(f"Reading from cache: {self._cache_path}")
            iterator = self._iter_from_cache()
            if self.shuffle:
                iterator = self._shuffle_buffer(iterator)
            for item in iterator:
                self._yielded_count += 1
                if self.log_interval > 0 and self._yielded_count % self.log_interval == 0:
                    logger.info(
                        f"[cache] Yielded {self._yielded_count} records, "
                        f"errors={self._error_count}"
                    )
                yield item
            logger.info(
                f"Cache iteration complete: yielded={self._yielded_count}, "
                f"errors={self._error_count}"
            )
            return

        # Stream from the configured source
        record_iterator = self.streamer.extract_structured_data()

        if self.shuffle:
            record_iterator = self._shuffle_buffer(record_iterator)

        cache_idx = 0
        for record in record_iterator:
            try:
                processed = self._process_record(record)
                self._yielded_count += 1

                # Write to cache if enabled
                if self._cache_enabled:
                    self._write_cache_record(processed, cache_idx)
                    cache_idx += 1

                if self.log_interval > 0 and self._yielded_count % self.log_interval == 0:
                    streamer_stats = self.streamer.stats()
                    logger.info(
                        f"[stream] Yielded {self._yielded_count} records, "
                        f"streamer: processed={streamer_stats['processed']}, "
                        f"errors={streamer_stats['errors']}, "
                        f"skipped={streamer_stats['skipped']}"
                    )

                yield processed
            except Exception as e:
                self._error_count += 1
                image_path = record.get('image_path', 'unknown')
                logger.warning(f"Failed to process record {image_path}: {e}")
                continue

        streamer_stats = self.streamer.stats()
        logger.info(
            f"Streaming iteration complete: yielded={self._yielded_count}, "
            f"errors={self._error_count}, "
            f"streamer: processed={streamer_stats['processed']}, "
            f"errors={streamer_stats['errors']}, "
            f"skipped={streamer_stats['skipped']}"
        )

    def _is_cache_populated(self) -> bool:
        """Check if the cache directory has at least one record."""
        if not self._cache_path or not os.path.isdir(self._cache_path):
            return False
        return os.path.exists(os.path.join(self._cache_path, "0.pt"))

    def __len__(self):
        """Get dataset length.

        For streaming datasets, exact length is unknown without reading
        entire archive. Returns 0 (unknown).
        """
        return 0

    def stats(self) -> Dict[str, int]:
        """Return iteration statistics."""
        return {
            'yielded': self._yielded_count,
            'errors': self._error_count,
            'streamer': self.streamer.stats(),
        }


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
        mean: Tuple[float, float, float] = (104.0, 117.0, 123.0),
        cache_dir: Optional[str] = None,
        max_retries: int = 3,
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
            cache_dir: Local directory for record caching (None to disable)
            max_retries: Maximum retries on network errors
        """
        config = StreamingConfig(
            data_source='hf_tar_stream',
            repo_id=repo_id,
            file_path=file_path,
            buffer_size=buffer_size,
            im_shape=im_shape,
            mean=mean,
            cache_dir=cache_dir,
            max_retries=max_retries,
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

    default_kwargs = {
        'batch_size': config.batch_size,
        'num_workers': 0,  # IterableDataset works best with num_workers=0
    }
    default_kwargs.update(loader_kwargs)

    return DataLoader(dataset, **default_kwargs)
