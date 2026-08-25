"""Hugging Face Tar Streamer - Stream and extract .tar.gz from HF Hub without local disk.

Uses HfFileSystem for remote file streaming and tarfile for in-memory extraction.
Includes retry logic for network errors, data integrity validation, and progress counters.
"""
import io
import struct
import tarfile
import json
import time
from typing import Iterator, Dict, Any, Optional, Tuple
from pathlib import Path
import logging

from huggingface_hub import HfFileSystem
from PIL import Image

logger = logging.getLogger(__name__)

# Minimal file-header magic bytes for integrity checks
_IMAGE_MAGIC = {
    b'\xff\xd8\xff': 'jpeg',
    b'\x89PNG': 'png',
    b'GIF8': 'gif',
    b'RIFF': 'bmp_or_webp',
}


def _detect_image_format(data: bytes) -> Optional[str]:
    """Detect image format from file header bytes. Returns format name or None."""
    for prefix, fmt in _IMAGE_MAGIC.items():
        if data[:len(prefix)] == prefix:
            return fmt
    return None


class HFTarStreamer:
    """Stream .tar.gz archives from Hugging Face Hub without downloading to disk.

    This class uses HfFileSystem to open a remote file stream and tarfile to
    extract members on-the-fly in memory. Includes retry logic for transient
    network failures and integrity checks on extracted members.
    """

    def __init__(self, repo_id: str, file_path: str,
                 buffer_size: int = 8192 * 1024, max_retries: int = 3):
        """Initialize the HF tar streamer.

        Args:
            repo_id: Hugging Face repository ID (e.g., 'username/dataset-name')
            file_path: Path to .tar.gz file within the repository
            buffer_size: Buffer size for streaming (default: 8MB)
            max_retries: Maximum number of retries on network errors
        """
        self.repo_id = repo_id
        self.file_path = file_path
        self.buffer_size = buffer_size
        self.max_retries = max_retries
        self.fs = HfFileSystem()

        # Construct full path for HfFileSystem
        self.full_path = f"hf://datasets/{repo_id}/{file_path}"

        # Progress counters
        self.processed_count = 0
        self.error_count = 0
        self.skipped_count = 0

        logger.info(f"Initialized HFTarStreamer for {self.full_path} (max_retries={max_retries})")

    def reset_counters(self):
        """Reset progress counters."""
        self.processed_count = 0
        self.error_count = 0
        self.skipped_count = 0

    def _retry_open(self):
        """Open the remote tar stream with retry logic for network errors.

        Yields:
            An open tarfile.TarFile object (used as a context manager by caller).

        Raises:
            RuntimeError: If all retries are exhausted.
        """
        last_err = None
        for attempt in range(1, self.max_retries + 1):
            try:
                remote_file = self.fs.open(self.full_path, 'rb')
                return remote_file
            except Exception as e:
                last_err = e
                wait = min(2 ** attempt, 30)
                logger.warning(
                    f"Attempt {attempt}/{self.max_retries} failed to open {self.full_path}: {e}. "
                    f"Retrying in {wait}s..."
                )
                time.sleep(wait)
        raise RuntimeError(
            f"Failed to open {self.full_path} after {self.max_retries} attempts: {last_err}"
        )

    @staticmethod
    def _validate_member(member_name: str, file_data: bytes) -> bool:
        """Basic integrity check on extracted tar member.

        Checks that the data is non-empty and, for known image formats,
        that the file header matches the extension.

        Args:
            member_name: Name of the tar member
            file_data: Raw bytes extracted from the tar

        Returns:
            True if the data looks valid, False otherwise.
        """
        if not file_data:
            logger.warning(f"Empty data for member {member_name}")
            return False

        suffix = Path(member_name).suffix.lower()
        if suffix in ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'):
            fmt = _detect_image_format(file_data)
            if fmt is None:
                logger.warning(
                    f"Image header mismatch for {member_name}: "
                    f"extension={suffix}, detected_header=none"
                )
                return False
        return True

    def _is_image_file(self, filename: str) -> bool:
        """Check if file is an image based on extension."""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
        return Path(filename).suffix.lower() in image_extensions

    def _is_json_file(self, filename: str) -> bool:
        """Check if file is a JSON file."""
        return Path(filename).suffix.lower() == '.json'

    def stream_tar_members(self) -> Iterator[Tuple[str, bytes, tarfile.TarInfo]]:
        """Stream tar archive members one by one with retry on network errors.

        Yields:
            Tuple of (member_name, file_data, tarinfo) for each member
        """
        remote_file = None
        try:
            remote_file = self._retry_open()
            with tarfile.open(fileobj=remote_file, mode='r|gz') as tar:
                logger.info(f"Successfully opened tar stream from {self.full_path}")

                for member in tar:
                    if not member.isfile():
                        continue

                    try:
                        file_obj = tar.extractfile(member)
                        if file_obj is None:
                            self.skipped_count += 1
                            continue

                        try:
                            file_data = file_obj.read()
                            if not self._validate_member(member.name, file_data):
                                self.skipped_count += 1
                                continue
                            self.processed_count += 1
                            yield member.name, file_data, member
                        finally:
                            file_obj.close()
                    except (tarfile.TarError, OSError) as e:
                        self.error_count += 1
                        logger.warning(f"Error reading member {member.name}: {e}")
                        continue

        except RuntimeError:
            raise
        except Exception as e:
            self.error_count += 1
            logger.error(f"Error streaming tar archive {self.full_path}: {e}")
            raise
        finally:
            if remote_file is not None:
                try:
                    remote_file.close()
                except Exception:
                    pass

    def extract_structured_data(self) -> Iterator[Dict[str, Any]]:
        """Extract and structure data from tar archive.

        Yields structured records containing:
        - image_path: Original path in archive
        - image: PIL Image object
        - annotation: Parsed JSON annotation (if available)
        - labels: List of attribute labels
        - Any other fields from JSON annotations

        This method handles nested directory structures and matches images
        with their corresponding JSON annotations.
        """
        self.reset_counters()
        image_cache: Dict[str, Image.Image] = {}
        annotation_cache: Dict[str, Dict[str, Any]] = {}

        for member_name, file_data, tar_info in self.stream_tar_members():
            normalized_path = member_name

            if self._is_json_file(member_name):
                try:
                    annotation = json.loads(file_data.decode('utf-8'))
                    annotation_cache[normalized_path] = annotation

                    image_path = annotation.get('image_path', '')
                    if image_path in image_cache:
                        record = self._create_record(
                            image_path, image_cache.pop(image_path), annotation
                        )
                        if record:
                            yield record

                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    self.error_count += 1
                    logger.warning(f"Failed to parse JSON from {member_name}: {e}")

            elif self._is_image_file(member_name):
                try:
                    image = Image.open(io.BytesIO(file_data)).convert('RGB')

                    base_name = Path(normalized_path).stem
                    potential_json_paths = [
                        str(Path(normalized_path).with_suffix('.json')),
                        f"{base_name}.json",
                    ]

                    matched = False
                    for json_path in potential_json_paths:
                        if json_path in annotation_cache:
                            annotation = annotation_cache.pop(json_path)
                            record = self._create_record(normalized_path, image, annotation)
                            if record:
                                yield record
                                matched = True
                                break

                    if not matched:
                        image_cache[normalized_path] = image

                        for anno_path, annotation in list(annotation_cache.items()):
                            anno_img_path = annotation.get('image_path', '')
                            if anno_img_path and (
                                anno_img_path in normalized_path or
                                normalized_path.endswith(anno_img_path)
                            ):
                                annotation_cache.pop(anno_path)
                                record = self._create_record(normalized_path, image, annotation)
                                if record:
                                    yield record
                                    image_cache.pop(normalized_path, None)
                                break

                except Exception as e:
                    self.error_count += 1
                    logger.warning(f"Failed to load image from {member_name}: {e}")

        # Yield remaining cached images without annotations
        for image_path, image in image_cache.items():
            record = self._create_record(image_path, image, None)
            if record:
                yield record

    def _create_record(self, image_path: str, image: Image.Image,
                       annotation: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Create a structured record from image and annotation.

        Args:
            image_path: Path to image in archive
            image: PIL Image object
            annotation: Parsed JSON annotation (optional)

        Returns:
            Structured record dictionary or None if invalid
        """
        record = {
            'image_path': image_path,
            'image': image,
        }

        if annotation:
            labels = set()
            if 'labels' in annotation:
                labels.update(annotation['labels'])
            elif 'attributes' in annotation:
                for _, attr_list in annotation['attributes'].items():
                    labels.update(attr_list)

            record['labels'] = list(labels)
            record['annotation'] = annotation

            for key in ['safe', 'label_vec', 'image_id']:
                if key in annotation:
                    record[key] = annotation[key]
        else:
            record['labels'] = []
            record['annotation'] = None

        return record

    def get_sample_count_estimate(self) -> Optional[int]:
        """Estimate number of samples (images) in archive.

        Note: This requires streaming through the entire archive once,
        so it's optional and should only be used if necessary.

        Returns:
            Estimated count or None if streaming fails
        """
        try:
            count = 0
            for member_name, _, _ in self.stream_tar_members():
                if self._is_image_file(member_name):
                    count += 1
            return count
        except Exception as e:
            logger.warning(f"Failed to estimate sample count: {e}")
            return None

    def stats(self) -> Dict[str, int]:
        """Return current progress counters."""
        return {
            'processed': self.processed_count,
            'errors': self.error_count,
            'skipped': self.skipped_count,
        }
