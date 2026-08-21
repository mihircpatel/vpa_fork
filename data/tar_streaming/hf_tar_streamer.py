"""Hugging Face Tar Streamer - Stream and extract .tar.gz from HF Hub without local disk.

Uses HfFileSystem for remote file streaming and tarfile for in-memory extraction.
"""
import io
import tarfile
import json
from typing import Iterator, Dict, Any, Optional, Tuple
from pathlib import Path
import logging

from huggingface_hub import HfFileSystem
from PIL import Image

logger = logging.getLogger(__name__)


class HFTarStreamer:
    """Stream .tar.gz archives from Hugging Face Hub without downloading to disk.

    This class uses HfFileSystem to open a remote file stream and tarfile to
    extract members on-the-fly in memory.
    """

    def __init__(self, repo_id: str, file_path: str, buffer_size: int = 8192 * 1024):
        """Initialize the HF tar streamer.

        Args:
            repo_id: Hugging Face repository ID (e.g., 'username/dataset-name')
            file_path: Path to .tar.gz file within the repository
            buffer_size: Buffer size for streaming (default: 8MB)
        """
        self.repo_id = repo_id
        self.file_path = file_path
        self.buffer_size = buffer_size
        self.fs = HfFileSystem()

        # Construct full path for HfFileSystem
        self.full_path = f"hf://datasets/{repo_id}/{file_path}"

        logger.info(f"Initialized HFTarStreamer for {self.full_path}")

    def _is_image_file(self, filename: str) -> bool:
        """Check if file is an image based on extension."""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
        return Path(filename).suffix.lower() in image_extensions

    def _is_json_file(self, filename: str) -> bool:
        """Check if file is a JSON file."""
        return Path(filename).suffix.lower() == '.json'

    def stream_tar_members(self) -> Iterator[Tuple[str, bytes, tarfile.TarInfo]]:
        """Stream tar archive members one by one.

        Yields:
            Tuple of (member_name, file_data, tarinfo) for each member
        """
        try:
            # Open remote file stream
            with self.fs.open(self.full_path, 'rb') as remote_file:
                # Open tar file from stream using streaming mode 'r|gz'
                # This mode reads sequentially without seeking, perfect for streams
                with tarfile.open(fileobj=remote_file, mode='r|gz') as tar:
                    logger.info(f"Successfully opened tar stream from {self.full_path}")

                    for member in tar:
                        # Skip directories
                        if not member.isfile():
                            continue

                        # Extract file data
                        file_obj = tar.extractfile(member)
                        if file_obj is None:
                            continue

                        try:
                            file_data = file_obj.read()
                            yield member.name, file_data, member
                        finally:
                            file_obj.close()

        except Exception as e:
            logger.error(f"Error streaming tar archive: {e}")
            raise

    def extract_structured_data(self) -> Iterator[Dict[str, Any]]:
        """Extract and structure data from tar archive.

        Yields structured records containing:
        - image_path: Original path in archive
        - image_data: PIL Image object
        - annotation: Parsed JSON annotation (if available)
        - labels: List of attribute labels
        - Any other fields from JSON annotations

        This method handles nested directory structures and matches images
        with their corresponding JSON annotations.
        """
        # Cache for matching images with annotations
        image_cache = {}
        annotation_cache = {}

        for member_name, file_data, tar_info in self.stream_tar_members():
            # Normalize path (remove leading directory components if needed)
            normalized_path = member_name

            if self._is_json_file(member_name):
                # Parse JSON annotation
                try:
                    annotation = json.loads(file_data.decode('utf-8'))
                    annotation_cache[normalized_path] = annotation

                    # Try to match with already-loaded image
                    image_path = annotation.get('image_path', '')
                    if image_path in image_cache:
                        record = self._create_record(
                            image_path,
                            image_cache.pop(image_path),
                            annotation
                        )
                        if record:
                            yield record

                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON from {member_name}: {e}")

            elif self._is_image_file(member_name):
                # Load image
                try:
                    image = Image.open(io.BytesIO(file_data)).convert('RGB')

                    # Try to find matching annotation
                    # Look for JSON with same base name
                    base_name = Path(normalized_path).stem
                    potential_json_paths = [
                        str(Path(normalized_path).with_suffix('.json')),
                        f"{base_name}.json",
                    ]

                    matched = False
                    for json_path in potential_json_paths:
                        if json_path in annotation_cache:
                            annotation = annotation_cache.pop(json_path)
                            record = self._create_record(
                                normalized_path,
                                image,
                                annotation
                            )
                            if record:
                                yield record
                                matched = True
                                break

                    if not matched:
                        # Cache image for later matching
                        image_cache[normalized_path] = image

                        # Also try to match by image_path in cached annotations
                        for anno_path, annotation in list(annotation_cache.items()):
                            anno_img_path = annotation.get('image_path', '')
                            if anno_img_path and (
                                anno_img_path in normalized_path or
                                normalized_path.endswith(anno_img_path)
                            ):
                                annotation_cache.pop(anno_path)
                                record = self._create_record(
                                    normalized_path,
                                    image,
                                    annotation
                                )
                                if record:
                                    yield record
                                    image_cache.pop(normalized_path, None)
                                    break

                except Exception as e:
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
            # Extract labels
            labels = set()
            if 'labels' in annotation:
                labels.update(annotation['labels'])
            elif 'attributes' in annotation:
                for _, attr_list in annotation['attributes'].items():
                    labels.update(attr_list)

            record['labels'] = list(labels)
            record['annotation'] = annotation

            # Copy other useful fields
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
