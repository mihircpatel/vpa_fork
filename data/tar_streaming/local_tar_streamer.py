"""Local Tar Streamer - Stream and extract .tar.gz archives from local disk.

Provides the same functionality as the HF streamers but reads archives from
local storage on the same machine instead of downloading from a remote
repository (e.g. Hugging Face Hub). This is useful when the dataset archive
is already present locally and the user wants to avoid an explicit extraction
step, or when they want to train from a tar archive without inflating it to a
directory tree on disk.

Supports two modes:
1. Combined archive mode: Images and annotations in a single .tar.gz
2. Dual archive mode: Separate .tar.gz files for images and annotations,
   optionally filtered by a .txt annotation list.

Includes data integrity validation and progress counters mirroring the
HFTarStreamer / HFDualArchiveStreamer APIs, so the dataset adapter can treat
local and remote sources uniformly.
"""
import io
import os.path as osp
import tarfile
import json
from typing import Iterator, Dict, Any, Optional, List, Tuple
from pathlib import Path
import logging

from PIL import Image

logger = logging.getLogger(__name__)


class LocalTarStreamer:
    """Stream .tar.gz archives from local storage without downloading.

    This class is a drop-in replacement for HFTarStreamer / HFDualArchiveStreamer
    that reads from local disk instead of a Hugging Face repository. It exposes
    the same `extract_structured_data()` / `stream_tar_members()` / `stats()`
    interface so callers can switch sources via configuration without changing
    their training or inference code.
    """

    def __init__(
        self,
        file_path: Optional[str] = None,
        image_archive_path: Optional[str] = None,
        annotation_archive_path: Optional[str] = None,
        anno_list_path: Optional[str] = None,
        anno_list_content: Optional[List[str]] = None,
        buffer_size: int = 8192 * 1024,
    ):
        """Initialize the local tar streamer.

        Combined archive mode:
            file_path=local_path.tar.gz

        Dual archive mode:
            image_archive_path=images.tar.gz
            annotation_archive_path=annotations.tar.gz
            anno_list_path=anno_list.txt (optional, local path)

        Args:
            file_path: Path to a combined .tar.gz file on local disk
            image_archive_path: Path to an image .tar.gz file on local disk
            annotation_archive_path: Path to an annotation .tar.gz file on local disk
            anno_list_path: Path to a .txt file listing annotations to load
            anno_list_content: Pre-loaded list of annotation paths (alternative to anno_list_path)
            buffer_size: Buffer size placeholder kept for API parity with HF streamer
        """
        self.file_path = file_path
        self.image_archive_path = image_archive_path
        self.annotation_archive_path = annotation_archive_path
        self.anno_list_path = anno_list_path
        self.buffer_size = buffer_size

        self._is_dual = self.image_archive_path is not None or \
            self.annotation_archive_path is not None or \
            self.anno_list_path is not None

        # Progress counters
        self.processed_count = 0
        self.error_count = 0
        self.skipped_count = 0

        # Load annotation list
        if anno_list_content is not None:
            self.anno_list = list(anno_list_content)
        elif self.anno_list_path:
            self.anno_list = self._load_anno_list(self.anno_list_path)
        else:
            self.anno_list = None

        if self._is_dual:
            logger.info(
                f"Initialized LocalTarStreamer (dual archive mode): "
                f"images={image_archive_path}, annotations={annotation_archive_path}, "
                f"anno_list={anno_list_path}, "
                f"num_annotations={len(self.anno_list) if self.anno_list else 'all'}"
            )
        else:
            logger.info(
                f"Initialized LocalTarStreamer (combined archive mode): "
                f"file={file_path}"
            )

    def reset_counters(self):
        """Reset progress counters."""
        self.processed_count = 0
        self.error_count = 0
        self.skipped_count = 0

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
            from .hf_tar_streamer import _detect_image_format
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

    def _load_anno_list(self, anno_list_path: str) -> List[str]:
        """Load annotation list from a local .txt file.

        Args:
            anno_list_path: Path to a local .txt file

        Returns:
            List of annotation paths

        Raises:
            FileNotFoundError: If the file cannot be read
        """
        try:
            with open(anno_list_path, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(lines)} annotations from local file: {anno_list_path}")
            return lines
        except Exception as e:
            logger.error(f"Failed to load annotation list from {anno_list_path}: {e}")
            raise FileNotFoundError(f"Could not load annotation list: {anno_list_path}")

    def _normalize_path(self, path: str) -> str:
        """Normalize path for matching.

        Extracts base filename for matching across different directory structures.
        Example: 'annotations/train2017/000000000001.json' -> '000000000001'
        """
        return Path(path).stem

    def _verify_archive_path(self, path: Optional[str]) -> None:
        """Verify a local archive path exists.

        Args:
            path: Local path to check

        Raises:
            FileNotFoundError: If the path does not exist
        """
        if not path:
            raise ValueError("Archive path is required")
        if not osp.exists(path):
            raise FileNotFoundError(f"Local archive not found: {path}")

    def _open_tar(self, path: str) -> tarfile.TarFile:
        """Open a local tar.gz archive.

        Args:
            path: Local .tar.gz path

        Returns:
            An open tarfile.TarFile
        """
        self._verify_archive_path(path)
        return tarfile.open(path, mode='r:gz')

    def stream_tar_members(self, path: str) -> Iterator[Tuple[str, bytes, tarfile.TarInfo]]:
        """Stream local tar archive members one by one.

        Args:
            path: Local .tar.gz path

        Yields:
            Tuple of (member_name, file_data, tarinfo) for each member
        """
        self._verify_archive_path(path)
        try:
            with tarfile.open(path, mode='r:gz') as tar:
                logger.info(f"Streaming local tar archive: {path}")

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
        except FileNotFoundError:
            raise
        except Exception as e:
            self.error_count += 1
            logger.error(f"Error streaming local tar archive {path}: {e}")
            raise

    def _stream_data_members(self, path: str) -> Iterator[Tuple[str, bytes]]:
        """Stream a local tar.gz archive, yielding (member_name, data) tuples."""
        for member_name, file_data, _ in self.stream_tar_members(path):
            yield member_name, file_data

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

    def extract_structured_data(self) -> Iterator[Dict[str, Any]]:
        """Extract and structure data from local tar archives.

        Automatically dispatches to combined or dual archive mode based on
        the constructor arguments.

        Yields:
            Structured records containing:
            - image: PIL Image object
            - image_path: Original image path
            - annotation: Parsed JSON annotation (if available)
            - labels: List of attribute labels
        """
        if self._is_dual:
            yield from self._extract_dual()
        else:
            yield from self._extract_combined()

    # -- Combined archive mode ------------------------------------------------

    def _extract_combined(self) -> Iterator[Dict[str, Any]]:
        """Extract records from a single combined local archive.

        Yields:
            Structured records from combined archive
        """
        self._verify_archive_path(self.file_path)
        self.reset_counters()
        image_cache: Dict[str, Image.Image] = {}
        annotation_cache: Dict[str, Dict[str, Any]] = {}

        for member_name, file_data, _ in self.stream_tar_members(self.file_path):
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

    # -- Dual archive mode ----------------------------------------------------

    def _extract_dual(self) -> Iterator[Dict[str, Any]]:
        """Extract records from separate image and annotation local archives.

        Yields:
            Structured records from dual archives
        """
        self._verify_archive_path(self.image_archive_path)
        self._verify_archive_path(self.annotation_archive_path)
        self.reset_counters()

        # Step 1: Build annotation index
        logger.info("Building annotation index...")
        anno_index = self._build_annotation_index()

        # Step 2: Filter by anno_list if provided
        if self.anno_list:
            anno_index = self._filter_annotation_index(anno_index)

        if not anno_index:
            logger.warning("No annotations found in index")
            return

        # Step 3: Stream images and match with annotations
        logger.info("Streaming images and matching with annotations...")
        matched_count = 0

        for member_name, file_data in self._stream_data_members(self.image_archive_path):
            if not self._is_image_file(member_name):
                continue

            normalized_name = self._normalize_path(member_name)

            if normalized_name not in anno_index:
                continue

            try:
                image = Image.open(io.BytesIO(file_data)).convert('RGB')
            except Exception as e:
                self.error_count += 1
                logger.warning(f"Failed to load image {member_name}: {e}")
                continue

            anno_data = anno_index[normalized_name]
            annotation = anno_data['annotation']

            labels = self._extract_labels(annotation)

            record = {
                'image': image,
                'image_path': annotation.get('image_path', member_name),
                'annotation': annotation,
                'labels': labels,
                'original_image_path': member_name,
                'original_anno_path': anno_data['original_path']
            }

            for key in ['safe', 'label_vec', 'image_id']:
                if key in annotation:
                    record[key] = annotation[key]

            matched_count += 1
            yield record

        logger.info(
            f"Matched {matched_count}/{len(anno_index)} annotations with images"
        )

    def _build_annotation_index(self) -> Dict[str, Dict[str, Any]]:
        """Build index of annotations by streaming the annotation archive.

        Returns:
            Dict mapping normalized filename to annotation data
        """
        anno_index = {}

        for member_name, file_data in self._stream_data_members(self.annotation_archive_path):
            if not member_name.endswith('.json'):
                continue

            try:
                annotation = json.loads(file_data.decode('utf-8'))
                normalized_name = self._normalize_path(member_name)

                anno_index[normalized_name] = {
                    'annotation': annotation,
                    'original_path': member_name
                }

            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                self.error_count += 1
                logger.warning(f"Failed to parse JSON {member_name}: {e}")
                continue

        logger.info(f"Built annotation index with {len(anno_index)} entries")
        return anno_index

    def _filter_annotation_index(
        self,
        anno_index: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Filter annotation index based on anno_list.

        Args:
            anno_index: Full annotation index

        Returns:
            Filtered annotation index containing only requested annotations
        """
        if self.anno_list is None:
            return anno_index

        filtered_index = {}
        for anno_path in self.anno_list:
            normalized_name = self._normalize_path(anno_path)
            if normalized_name in anno_index:
                filtered_index[normalized_name] = anno_index[normalized_name]
            else:
                logger.warning(f"Annotation not found in archive: {anno_path}")

        logger.info(
            f"Filtered annotation index: {len(filtered_index)}/{len(anno_index)} annotations"
        )
        return filtered_index

    def _extract_labels(self, annotation: Dict[str, Any]) -> List[str]:
        """Extract labels from annotation.

        Args:
            annotation: Parsed JSON annotation

        Returns:
            List of label strings
        """
        labels = set()

        if 'labels' in annotation:
            labels.update(annotation['labels'])
        elif 'attributes' in annotation:
            for _, attr_list in annotation['attributes'].items():
                labels.update(attr_list)

        return list(labels)

    def get_sample_count(self) -> int:
        """Get number of samples.

        Returns:
            Number of samples (from anno_list if available, else -1)
        """
        if self.anno_list:
            return len(self.anno_list)
        return -1

    def stats(self) -> Dict[str, int]:
        """Return current progress counters."""
        return {
            'processed': self.processed_count,
            'errors': self.error_count,
            'skipped': self.skipped_count,
        }
