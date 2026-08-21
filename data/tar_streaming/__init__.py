"""HF Tar Streaming Module - Stream datasets from Hugging Face Hub.

This module provides utilities for streaming .tar.gz archives from Hugging Face
repositories without downloading to local disk, ideal for disk-constrained
environments like Google Colab.

Example usage:
    >>> from data.tar_streaming import StreamingConfig, StreamingPAPDataset
    >>> config = StreamingConfig(
    ...     data_source='hf_tar_stream',
    ...     repo_id='username/dataset-name',
    ...     file_path='train.tar.gz',
    ...     buffer_size=1000
    ... )
    >>> dataset = StreamingPAPDataset(config=config, shuffle=True)
    >>> loader = DataLoader(dataset, batch_size=32)
"""

from .config import StreamingConfig
from .hf_tar_streamer import HFTarStreamer
from .dual_archive_streamer import HFDualArchiveStreamer
from .streaming_dataset import (
    StreamingPAPDataset,
    StreamingPAPDatasetFromFile,
    get_streaming_dataloader
)

__all__ = [
    'StreamingConfig',
    'HFTarStreamer',
    'HFDualArchiveStreamer',
    'StreamingPAPDataset',
    'StreamingPAPDatasetFromFile',
    'get_streaming_dataloader',
]

__version__ = '1.0.0'
