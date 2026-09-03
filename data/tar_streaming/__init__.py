"""Tar Streaming Module - Stream datasets from Hugging Face Hub or local disk.

This module provides utilities for streaming .tar.gz archives without 
inflating them to a directory tree on disk, ideal for disk-constrained
environments like Google Colab. Data can be streamed from Hugging Face Hub
(remote, using HfFileSystem) or from local storage on the same machine.

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

Local streaming (data already on disk):
    >>> config = StreamingConfig(
    ...     data_source='local_tar_stream',
    ...     file_path='./datasets/train.tar.gz',
    ...     buffer_size=1000
    ... )
    >>> dataset = StreamingPAPDataset(config=config, shuffle=True)
"""

from .config import StreamingConfig
from .hf_tar_streamer import HFTarStreamer
from .dual_archive_streamer import HFDualArchiveStreamer
from .local_tar_streamer import LocalTarStreamer
from .streaming_dataset import (
    StreamingPAPDataset,
    StreamingPAPDatasetFromFile,
    get_streaming_dataloader
)

__all__ = [
    'StreamingConfig',
    'HFTarStreamer',
    'HFDualArchiveStreamer',
    'LocalTarStreamer',
    'StreamingPAPDataset',
    'StreamingPAPDatasetFromFile',
    'get_streaming_dataloader',
]

__version__ = '1.1.0'
