"""Dual Archive Streamer - Handle separate image and annotation tar archives.

This module supports the common dataset pattern where images and annotations
are stored in separate .tar.gz files, with a .txt file listing the annotations
to load (similar to local PAPDataset behavior).
"""
import io
import tarfile
import json
import os.path as osp
from typing import Iterator, Dict, Any, Optional, List, Tuple
from pathlib import Path
import logging

from huggingface_hub import HfFileSystem
from PIL import Image
from vispr import DS_ROOT

logger = logging.getLogger(__name__)


class HFDualArchiveStreamer:
    """Stream from separate image and annotation archives on HuggingFace Hub.

    This class handles the pattern where:
    - Images are in one .tar.gz archive (e.g., train2017_images.tar.gz)
    - Annotations are in another .tar.gz archive (e.g., train2017_annotations.tar.gz)
    - A .txt file lists which annotations to load (e.g., train2017.txt)

    Archive structure expected:
    - train2017/000000000001.jpg (in image archive)
    - train2017/000000000001.json (in annotation archive)
    """

    def __init__(
        self,
        repo_id: str,
        image_archive_path: str,
        annotation_archive_path: str,
        anno_list_path: Optional[str] = None,
        anno_list_content: Optional[List[str]] = None
    ):
        """Initialize dual archive streamer.

        Args:
            repo_id: HuggingFace repository ID
            image_archive_path: Path to image .tar.gz in repo
            annotation_archive_path: Path to annotation .tar.gz in repo
            anno_list_path: Path to .txt file listing annotations (in repo or local)
            anno_list_content: Pre-loaded list of annotation paths (alternative to anno_list_path)
        """
        self.repo_id = repo_id
        self.image_archive_path = image_archive_path
        self.annotation_archive_path = annotation_archive_path
        self.anno_list_path = anno_list_path
        self.fs = HfFileSystem()

        # Load annotation list
        if anno_list_content is not None:
            self.anno_list = anno_list_content
        elif anno_list_path:
            self.anno_list = self._load_anno_list(anno_list_path)
        else:
            # If no list provided, will stream all annotations found
            self.anno_list = None

        logger.info(
            f"Initialized HFDualArchiveStreamer: "
            f"images={image_archive_path}, annotations={annotation_archive_path}, "
            f"num_annotations={len(self.anno_list) if self.anno_list else 'all'}"
        )

    def _load_anno_list(self, anno_list_path: str) -> List[str]:
        """Load annotation list from file.

        Args:
            anno_list_path: Path to .txt file (local or on HF Hub)

        Returns:
            List of annotation paths
        """
        anno_list_loaded = False
        # Try loading from HF Hub first
        try:
            hf_path = f"hf://datasets/{self.repo_id}/{anno_list_path}"
            with self.fs.open(hf_path, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(lines)} annotations from HF Hub: {hf_path}")
            anno_list_loaded = True
            return lines
        except Exception as e:
            logger.debug(f"Could not load from HF Hub: {e}, trying local path")

        # Fall back to local file
        if not anno_list_loaded:
            try:
                anno_list_path = osp.join(DS_ROOT, anno_list_path)
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

    def _stream_tar_archive(self, archive_path: str) -> Iterator[Tuple[str, bytes]]:
        """Stream a single tar.gz archive from HF Hub.

        Args:
            archive_path: Path to .tar.gz file in repository

        Yields:
            Tuples of (member_name, file_data)
        """
        full_path = f"hf://datasets/{self.repo_id}/{archive_path}"

        try:
            with self.fs.open(full_path, 'rb') as remote_file:
                with tarfile.open(fileobj=remote_file, mode='r|gz') as tar:
                    logger.info(f"Streaming tar archive: {full_path}")

                    for member in tar:
                        if not member.isfile():
                            continue

                        file_obj = tar.extractfile(member)
                        if file_obj is None:
                            continue

                        try:
                            file_data = file_obj.read()
                            yield member.name, file_data
                        finally:
                            file_obj.close()

        except Exception as e:
            logger.error(f"Error streaming tar archive {full_path}: {e}")
            raise

    def _build_annotation_index(self) -> Dict[str, Dict[str, Any]]:
        """Build index of annotations by streaming annotation archive.

        Returns:
            Dict mapping normalized filename to annotation data
        """
        anno_index = {}

        for member_name, file_data in self._stream_tar_archive(self.annotation_archive_path):
            if not member_name.endswith('.json'):
                continue

            # Parse JSON
            try:
                annotation = json.loads(file_data.decode('utf-8'))
                normalized_name = self._normalize_path(member_name)

                # Store with both original path and normalized name
                anno_index[normalized_name] = {
                    'annotation': annotation,
                    'original_path': member_name
                }

            except json.JSONDecodeError as e:
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

    def extract_structured_data(self) -> Iterator[Dict[str, Any]]:
        """Extract and yield structured records from dual archives.

        Yields:
            Structured records containing:
            - image: PIL Image object
            - image_path: Original image path
            - annotation: Parsed JSON annotation
            - labels: List of attribute labels
        """
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

        for member_name, file_data in self._stream_tar_archive(self.image_archive_path):
            # Check if it's an image file
            if not self._is_image_file(member_name):
                continue

            # Get normalized name for matching
            normalized_name = self._normalize_path(member_name)

            # Check if we have annotation for this image
            if normalized_name not in anno_index:
                continue

            # Load image
            try:
                image = Image.open(io.BytesIO(file_data)).convert('RGB')
            except Exception as e:
                logger.warning(f"Failed to load image {member_name}: {e}")
                continue

            # Get annotation data
            anno_data = anno_index[normalized_name]
            annotation = anno_data['annotation']

            # Extract labels
            labels = self._extract_labels(annotation)

            # Create structured record
            record = {
                'image': image,
                'image_path': annotation.get('image_path', member_name),
                'annotation': annotation,
                'labels': labels,
                'original_image_path': member_name,
                'original_anno_path': anno_data['original_path']
            }

            # Add optional fields
            for key in ['safe', 'label_vec', 'image_id']:
                if key in annotation:
                    record[key] = annotation[key]

            matched_count += 1
            yield record

        logger.info(
            f"Matched {matched_count}/{len(anno_index)} annotations with images"
        )

    def _is_image_file(self, filename: str) -> bool:
        """Check if file is an image."""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
        return Path(filename).suffix.lower() in image_extensions

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
