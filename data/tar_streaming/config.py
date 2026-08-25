"""Configuration module for HF Tar Streaming data loader.

Handles configuration from YAML files or command-line arguments.
"""
import hashlib
import os
from dataclasses import dataclass, field
from typing import Optional, Tuple
import yaml
import os.path as osp


@dataclass
class StreamingConfig:
    """Configuration for streaming tar.gz datasets from Hugging Face Hub.

    Supports two modes:
    1. Combined archive mode: Images and annotations in single .tar.gz
    2. Dual archive mode: Separate .tar.gz files for images and annotations

    Attributes:
        data_source: Source type - 'local' or 'hf_tar_stream'
        repo_id: Hugging Face repository ID (e.g., 'username/dataset-name')

        # Combined archive mode (original)
        file_path: Path to combined .tar.gz file within the repository

        # Dual archive mode (new)
        image_archive_path: Path to image .tar.gz file
        annotation_archive_path: Path to annotation .tar.gz file
        anno_list_path: Path to .txt file listing annotations to load

        # Common settings
        buffer_size: Number of samples to keep in memory for shuffling
        im_shape: Image shape as (height, width) tuple
        mean: Mean values for normalization [B, G, R]
        num_classes: Number of attribute classes
        batch_size: Batch size for data loading
        num_workers: Number of worker processes for data loading

        # Streaming tuning
        chunk_size: Read chunk size in bytes for tar streaming (default 8MB)
        log_interval: Log progress every N records (0 to disable)
        max_retries: Max retries on network errors before giving up
        cache_dir: Local directory for record-level caching (None to disable)
    """
    data_source: str = "local"
    repo_id: Optional[str] = None

    # Combined archive mode
    file_path: Optional[str] = None

    # Dual archive mode
    image_archive_path: Optional[str] = None
    annotation_archive_path: Optional[str] = None
    anno_list_path: Optional[str] = None

    # Common settings
    buffer_size: int = 1000
    im_shape: Tuple[int, int] = (224, 224)
    mean: Tuple[float, float, float] = (104.0, 117.0, 123.0)
    num_classes: int = 68
    batch_size: int = 32
    num_workers: int = 2

    # Streaming tuning
    chunk_size: int = 8 * 1024 * 1024  # 8 MB
    log_interval: int = 100
    max_retries: int = 3
    cache_dir: Optional[str] = None

    def get_cache_key(self) -> str:
        """Derive a stable directory name for local record caching."""
        raw = f"{self.repo_id}:{self.file_path or ''}:{self.image_archive_path or ''}:{self.annotation_archive_path or ''}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def get_cache_path(self) -> Optional[str]:
        """Return the resolved cache directory path, or None if caching is off."""
        if not self.cache_dir:
            return None
        return osp.join(self.cache_dir, self.get_cache_key())

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'StreamingConfig':
        """Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            StreamingConfig instance
        """
        if not osp.exists(yaml_path):
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Convert nested dicts to flat structure
        flat_config = {}
        if 'streaming' in config_dict:
            flat_config.update(config_dict['streaming'])
        if 'data' in config_dict:
            flat_config.update(config_dict['data'])

        # Handle top-level keys
        for key in ['data_source', 'repo_id', 'file_path', 'buffer_size',
                    'im_shape', 'mean', 'num_classes', 'batch_size', 'num_workers',
                    'image_archive_path', 'annotation_archive_path', 'anno_list_path',
                    'chunk_size', 'log_interval', 'max_retries', 'cache_dir']:
            if key in config_dict:
                flat_config[key] = config_dict[key]

        # Convert lists to tuples for im_shape and mean
        if 'im_shape' in flat_config and isinstance(flat_config['im_shape'], list):
            flat_config['im_shape'] = tuple(flat_config['im_shape'])
        if 'mean' in flat_config and isinstance(flat_config['mean'], list):
            flat_config['mean'] = tuple(flat_config['mean'])

        return cls(**flat_config)

    @classmethod
    def from_args(cls, args) -> 'StreamingConfig':
        """Create configuration from command-line arguments.

        Args:
            args: argparse.Namespace with command-line arguments

        Returns:
            StreamingConfig instance
        """
        config = cls()

        # Update from args if present
        if hasattr(args, 'data_source') and args.data_source:
            config.data_source = args.data_source
        if hasattr(args, 'hf_repo') and args.hf_repo:
            config.repo_id = args.hf_repo

        # Combined archive mode
        if hasattr(args, 'hf_file_path') and args.hf_file_path:
            config.file_path = args.hf_file_path

        # Dual archive mode
        if hasattr(args, 'hf_image_archive') and args.hf_image_archive:
            config.image_archive_path = args.hf_image_archive
        if hasattr(args, 'hf_anno_archive') and args.hf_anno_archive:
            config.annotation_archive_path = args.hf_anno_archive
        if hasattr(args, 'hf_anno_list') and args.hf_anno_list:
            config.anno_list_path = args.hf_anno_list

        # Common settings
        if hasattr(args, 'buffer_size') and args.buffer_size:
            config.buffer_size = args.buffer_size
        if hasattr(args, 'batch_size') and args.batch_size:
            config.batch_size = args.batch_size
        if hasattr(args, 'num_classes') and args.num_classes:
            config.num_classes = args.num_classes

        # Streaming tuning
        if hasattr(args, 'chunk_size') and args.chunk_size:
            config.chunk_size = args.chunk_size
        if hasattr(args, 'log_interval') and args.log_interval is not None:
            config.log_interval = args.log_interval
        if hasattr(args, 'max_retries') and args.max_retries is not None:
            config.max_retries = args.max_retries
        if hasattr(args, 'cache_dir') and args.cache_dir:
            config.cache_dir = args.cache_dir

        return config

    def is_dual_archive_mode(self) -> bool:
        """Check if using dual archive mode (separate image/annotation archives).

        Returns:
            True if dual archive mode, False if combined archive mode
        """
        return (self.image_archive_path is not None or
                self.annotation_archive_path is not None or
                self.anno_list_path is not None)

    def validate(self) -> None:
        """Validate configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        if self.data_source not in ['local', 'hf_tar_stream']:
            raise ValueError(f"Invalid data_source: {self.data_source}. Must be 'local' or 'hf_tar_stream'")

        if self.data_source == 'hf_tar_stream':
            if not self.repo_id:
                raise ValueError("repo_id is required when data_source is 'hf_tar_stream'")

            # Check if using dual archive mode or combined archive mode
            is_dual_mode = self.is_dual_archive_mode()

            if is_dual_mode:
                # Dual archive mode validation
                if not self.image_archive_path:
                    raise ValueError("image_archive_path is required for dual archive mode")
                if not self.annotation_archive_path:
                    raise ValueError("annotation_archive_path is required for dual archive mode")
                if not self.anno_list_path:
                    raise ValueError("anno_list_path is required for dual archive mode")

                # Validate archive paths
                if not self.image_archive_path.endswith('.tar.gz'):
                    raise ValueError(f"image_archive_path must end with '.tar.gz', got: {self.image_archive_path}")
                if not self.annotation_archive_path.endswith('.tar.gz'):
                    raise ValueError(f"annotation_archive_path must end with '.tar.gz', got: {self.annotation_archive_path}")
            else:
                # Combined archive mode validation
                if not self.file_path:
                    raise ValueError("file_path is required for combined archive mode (or use dual archive mode)")
                if not self.file_path.endswith('.tar.gz'):
                    raise ValueError(f"file_path must end with '.tar.gz', got: {self.file_path}")

        if self.buffer_size <= 0:
            raise ValueError(f"buffer_size must be positive, got: {self.buffer_size}")

        if len(self.im_shape) != 2:
            raise ValueError(f"im_shape must be (height, width), got: {self.im_shape}")

        if len(self.mean) != 3:
            raise ValueError(f"mean must have 3 values [B, G, R], got: {self.mean}")

        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got: {self.chunk_size}")

        if self.log_interval < 0:
            raise ValueError(f"log_interval must be non-negative, got: {self.log_interval}")

        if self.max_retries < 0:
            raise ValueError(f"max_retries must be non-negative, got: {self.max_retries}")
