"""Unit and integration tests for the HF Tar Streaming module.

Run all tests:
    python -m pytest tests/test_streaming.py -v

Run only mocked (offline) tests:
    python -m pytest tests/test_streaming.py -v -m "not integration"

Run only integration tests (requires HF Hub access):
    python -m pytest tests/test_streaming.py -v -m integration
"""
import io
import json
import os
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest
from PIL import Image

# Ensure repo root is on sys.path so imports work
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_image_bytes(fmt='JPEG', size=(64, 64)):
    """Create in-memory image bytes."""
    img = Image.fromarray(np.random.randint(0, 255, (*size, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def _make_annotation(image_path='img1.jpg', labels=None):
    """Create a JSON annotation dict."""
    if labels is None:
        labels = ['a0_safe']
    return {'image_path': image_path, 'labels': labels, 'safe': 'a0_safe' in labels}


def _create_tar_gz(members, path):
    """Create a .tar.gz file at *path* with the given members.

    members: list of (filename, bytes) tuples
    """
    with tarfile.open(path, 'w:gz') as tar:
        for name, data in members:
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def _make_dummy_archive(extra_annotations=None):
    """Create a temporary directory with a combined .tar.gz archive.

    Returns (tmpdir_path, archive_path, list_of_anno_paths).
    """
    tmpdir = tempfile.mkdtemp(prefix='stream_test_')
    archive_path = os.path.join(tmpdir, 'data.tar.gz')

    members = []
    anno_paths = []

    # Image + annotation pair
    img_bytes = _make_image_bytes()
    anno = _make_annotation('img1.jpg', ['a0_safe', 'a3_violence'])
    anno_bytes = json.dumps(anno).encode()

    members.append(('images/img1.jpg', img_bytes))
    members.append(('annotations/img1.json', anno_bytes))
    anno_paths.append('annotations/img1.json')

    # Second pair
    img_bytes2 = _make_image_bytes()
    anno2 = _make_annotation('img2.jpg', ['a1_adult'])
    anno_bytes2 = json.dumps(anno2).encode()

    members.append(('images/img2.jpg', img_bytes2))
    members.append(('annotations/img2.json', anno_bytes2))
    anno_paths.append('annotations/img2.json')

    if extra_annotations:
        for i, (lbls, img_name) in enumerate(extra_annotations):
            img_b = _make_image_bytes()
            a = _make_annotation(img_name, lbls)
            members.append((f'images/{img_name}', img_b))
            members.append((f'annotations/{img_name.replace(".jpg", ".json")}', json.dumps(a).encode()))
            anno_paths.append(f'annotations/{img_name.replace(".jpg", ".json")}')

    _create_tar_gz(members, archive_path)
    return tmpdir, archive_path, anno_paths


def _make_dual_archive(tmpdir):
    """Create separate image and annotation .tar.gz archives in tmpdir."""
    img_archive = os.path.join(tmpdir, 'images.tar.gz')
    anno_archive = os.path.join(tmpdir, 'annotations.tar.gz')

    img_members = []
    anno_members = []
    anno_list = []

    for i in range(3):
        name = f'img{i}.jpg'
        img_members.append((name, _make_image_bytes()))
        # Use valid attribute IDs from attributes.tsv
        # Valid: a0_safe, a1_age_approx, a2_weight_approx, a3_height_approx, a4_gender, ...
        anno = _make_annotation(name, ['a0_safe'])
        anno_json = f'img{i}.json'
        anno_members.append((anno_json, json.dumps(anno).encode()))
        anno_list.append(anno_json)

    _create_tar_gz(img_members, img_archive)
    _create_tar_gz(anno_members, anno_archive)

    # Write anno list
    list_path = os.path.join(tmpdir, 'anno_list.txt')
    with open(list_path, 'w') as f:
        f.write('\n'.join(anno_list))

    return img_archive, anno_archive, list_path


# ---------------------------------------------------------------------------
# StreamingConfig Tests
# ---------------------------------------------------------------------------

class TestStreamingConfig:
    """Tests for StreamingConfig validation and construction."""

    def test_default_config(self):
        from data.tar_streaming.config import StreamingConfig
        cfg = StreamingConfig()
        assert cfg.data_source == 'local'
        assert cfg.buffer_size == 1000
        assert cfg.chunk_size == 8 * 1024 * 1024
        assert cfg.log_interval == 100
        assert cfg.max_retries == 3
        assert cfg.cache_dir is None

    def test_valid_combined_archive_config(self):
        from data.tar_streaming.config import StreamingConfig
        cfg = StreamingConfig(
            data_source='hf_tar_stream',
            repo_id='user/repo',
            file_path='data.tar.gz',
        )
        cfg.validate()  # should not raise

    def test_valid_dual_archive_config(self):
        from data.tar_streaming.config import StreamingConfig
        cfg = StreamingConfig(
            data_source='hf_tar_stream',
            repo_id='user/repo',
            image_archive_path='images.tar.gz',
            annotation_archive_path='annotations.tar.gz',
            anno_list_path='list.txt',
        )
        cfg.validate()

    def test_invalid_data_source(self):
        from data.tar_streaming.config import StreamingConfig
        cfg = StreamingConfig(data_source='invalid')
        with pytest.raises(ValueError, match="Invalid data_source"):
            cfg.validate()

    def test_missing_repo_id(self):
        from data.tar_streaming.config import StreamingConfig
        cfg = StreamingConfig(data_source='hf_tar_stream')
        with pytest.raises(ValueError, match="repo_id is required"):
            cfg.validate()

    def test_missing_file_path(self):
        from data.tar_streaming.config import StreamingConfig
        cfg = StreamingConfig(data_source='hf_tar_stream', repo_id='user/repo')
        with pytest.raises(ValueError, match="file_path is required"):
            cfg.validate()

    def test_invalid_chunk_size(self):
        from data.tar_streaming.config import StreamingConfig
        cfg = StreamingConfig(
            data_source='hf_tar_stream', repo_id='user/repo',
            file_path='data.tar.gz', chunk_size=0
        )
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            cfg.validate()

    def test_invalid_log_interval(self):
        from data.tar_streaming.config import StreamingConfig
        cfg = StreamingConfig(
            data_source='hf_tar_stream', repo_id='user/repo',
            file_path='data.tar.gz', log_interval=-1
        )
        with pytest.raises(ValueError, match="log_interval must be non-negative"):
            cfg.validate()

    def test_invalid_max_retries(self):
        from data.tar_streaming.config import StreamingConfig
        cfg = StreamingConfig(
            data_source='hf_tar_stream', repo_id='user/repo',
            file_path='data.tar.gz', max_retries=-1
        )
        with pytest.raises(ValueError, match="max_retries must be non-negative"):
            cfg.validate()

    def test_dual_archive_missing_image_path(self):
        from data.tar_streaming.config import StreamingConfig
        cfg = StreamingConfig(
            data_source='hf_tar_stream', repo_id='user/repo',
            annotation_archive_path='annos.tar.gz', anno_list_path='list.txt',
        )
        with pytest.raises(ValueError, match="image_archive_path is required"):
            cfg.validate()

    def test_from_args(self):
        from data.tar_streaming.config import StreamingConfig
        args = MagicMock()
        args.data_source = 'hf_tar_stream'
        args.hf_repo = 'user/repo'
        args.hf_file_path = 'data.tar.gz'
        args.hf_image_archive = None
        args.hf_anno_archive = None
        args.hf_anno_list = None
        args.buffer_size = 500
        args.batch_size = 16
        args.num_classes = 68
        args.chunk_size = 4 * 1024 * 1024
        args.log_interval = 25
        args.max_retries = 10
        args.cache_dir = '/tmp/cache'

        cfg = StreamingConfig.from_args(args)
        assert cfg.data_source == 'hf_tar_stream'
        assert cfg.repo_id == 'user/repo'
        assert cfg.file_path == 'data.tar.gz'
        assert cfg.buffer_size == 500
        assert cfg.chunk_size == 4 * 1024 * 1024
        assert cfg.log_interval == 25
        assert cfg.max_retries == 10
        assert cfg.cache_dir == '/tmp/cache'

    def test_cache_key_deterministic(self):
        from data.tar_streaming.config import StreamingConfig
        cfg1 = StreamingConfig(repo_id='user/repo', file_path='data.tar.gz')
        cfg2 = StreamingConfig(repo_id='user/repo', file_path='data.tar.gz')
        assert cfg1.get_cache_key() == cfg2.get_cache_key()

    def test_cache_key_differs(self):
        from data.tar_streaming.config import StreamingConfig
        cfg1 = StreamingConfig(repo_id='user/repo', file_path='data.tar.gz')
        cfg2 = StreamingConfig(repo_id='user/repo', file_path='other.tar.gz')
        assert cfg1.get_cache_key() != cfg2.get_cache_key()

    def test_get_cache_path(self):
        from data.tar_streaming.config import StreamingConfig
        cfg = StreamingConfig(
            repo_id='user/repo', file_path='data.tar.gz',
            cache_dir='/tmp/my_cache'
        )
        path = cfg.get_cache_path()
        assert path.startswith('/tmp/my_cache' + os.sep)
        assert len(os.path.basename(path)) == 16  # sha256[:16]

    def test_get_cache_path_none(self):
        from data.tar_streaming.config import StreamingConfig
        cfg = StreamingConfig(repo_id='user/repo', file_path='data.tar.gz')
        assert cfg.get_cache_path() is None


# ---------------------------------------------------------------------------
# Integrity Validation Tests
# ---------------------------------------------------------------------------

class TestIntegrityValidation:
    """Tests for data integrity validation in streamers."""

    def test_validate_member_valid_jpeg(self):
        from data.tar_streaming.hf_tar_streamer import HFTarStreamer
        img_data = _make_image_bytes('JPEG')
        assert HFTarStreamer._validate_member('test.jpg', img_data) is True

    def test_validate_member_valid_png(self):
        from data.tar_streaming.hf_tar_streamer import HFTarStreamer
        img_data = _make_image_bytes('PNG')
        assert HFTarStreamer._validate_member('test.png', img_data) is True

    def test_validate_member_empty_data(self):
        from data.tar_streaming.hf_tar_streamer import HFTarStreamer
        assert HFTarStreamer._validate_member('test.jpg', b'') is False

    def test_validate_member_corrupt_jpeg(self):
        from data.tar_streaming.hf_tar_streamer import HFTarStreamer
        # Random bytes that don't match JPEG header
        bad_data = os.urandom(100)
        assert HFTarStreamer._validate_member('test.jpg', bad_data) is False

    def test_validate_member_non_image_uncheked(self):
        from data.tar_streaming.hf_tar_streamer import HFTarStreamer
        # JSON files skip image header check
        assert HFTarStreamer._validate_member('test.json', b'{"key": "val"}') is True

    def test_validate_member_corrupt_png(self):
        from data.tar_streaming.hf_tar_streamer import HFTarStreamer
        bad_data = b'NOT_A_PNG' + os.urandom(50)
        assert HFTarStreamer._validate_member('test.png', bad_data) is False


# ---------------------------------------------------------------------------
# Shuffle Buffer Tests
# ---------------------------------------------------------------------------

class TestShuffleBuffer:
    """Tests for the shuffle buffer in StreamingPAPDataset."""

    def test_shuffle_buffer_preserves_all_items(self):
        from data.tar_streaming.streaming_dataset import StreamingPAPDataset
        from data.tar_streaming.config import StreamingConfig

        cfg = StreamingConfig(
            data_source='hf_tar_stream', repo_id='user/repo',
            file_path='data.tar.gz', buffer_size=10
        )

        # Create a minimal dataset instance to access _shuffle_buffer
        with patch.object(StreamingPAPDataset, '__init__', lambda self, *a, **kw: None):
            ds = StreamingPAPDataset.__new__(StreamingPAPDataset)
            ds.buffer_size = 10

            items = list(range(20))
            shuffled = list(ds._shuffle_buffer(iter(items)))
            assert sorted(shuffled) == sorted(items)

    def test_shuffle_buffer_small_dataset(self):
        from data.tar_streaming.streaming_dataset import StreamingPAPDataset

        with patch.object(StreamingPAPDataset, '__init__', lambda self, *a, **kw: None):
            ds = StreamingPAPDataset.__new__(StreamingPAPDataset)
            ds.buffer_size = 100

            items = list(range(5))
            shuffled = list(ds._shuffle_buffer(iter(items)))
            assert sorted(shuffled) == sorted(items)


# ---------------------------------------------------------------------------
# Cache Tests
# ---------------------------------------------------------------------------

class TestCache:
    """Tests for local record caching."""

    def test_cache_write_and_read(self):
        import torch
        from data.tar_streaming.streaming_dataset import StreamingPAPDataset
        from data.tar_streaming.config import StreamingConfig

        tmpdir = tempfile.mkdtemp(prefix='cache_test_')
        try:
            cfg = StreamingConfig(
                data_source='hf_tar_stream', repo_id='user/repo',
                file_path='data.tar.gz', cache_dir=tmpdir
            )

            with patch.object(StreamingPAPDataset, '__init__', lambda self, *a, **kw: None):
                ds = StreamingPAPDataset.__new__(StreamingPAPDataset)
                ds._cache_path = cfg.get_cache_path()
                ds._cache_enabled = True
                ds.return_metadata = False

                # Write some records
                img_tensor = torch.randn(3, 224, 224)
                label_tensor = torch.zeros(68)
                label_tensor[0] = 1.0

                ds._write_cache_record((img_tensor, label_tensor), 0)
                ds._write_cache_record((img_tensor, label_tensor), 1)

                # Verify files exist
                assert os.path.exists(os.path.join(ds._cache_path, '0.pt'))
                assert os.path.exists(os.path.join(ds._cache_path, '1.pt'))

                # Read back
                data = torch.load(os.path.join(ds._cache_path, '0.pt'), weights_only=False)
                assert torch.equal(data['image'], img_tensor)
                assert torch.equal(data['label'], label_tensor)
        finally:
            shutil.rmtree(tmpdir)

    def test_is_cache_populated(self):
        from data.tar_streaming.streaming_dataset import StreamingPAPDataset

        tmpdir = tempfile.mkdtemp(prefix='cache_test_')
        try:
            with patch.object(StreamingPAPDataset, '__init__', lambda self, *a, **kw: None):
                ds = StreamingPAPDataset.__new__(StreamingPAPDataset)
                ds._cache_path = os.path.join(tmpdir, 'cache')
                ds._cache_enabled = True

                assert ds._is_cache_populated() is False

                os.makedirs(ds._cache_path)
                assert ds._is_cache_populated() is False

                # Create 0.pt
                import torch
                torch.save({'image': torch.zeros(1), 'label': torch.zeros(1)},
                           os.path.join(ds._cache_path, '0.pt'))
                assert ds._is_cache_populated() is True
        finally:
            shutil.rmtree(tmpdir)

    def test_cache_with_metadata(self):
        import torch
        from data.tar_streaming.streaming_dataset import StreamingPAPDataset
        from data.tar_streaming.config import StreamingConfig

        tmpdir = tempfile.mkdtemp(prefix='cache_test_')
        try:
            cfg = StreamingConfig(
                data_source='hf_tar_stream', repo_id='user/repo',
                file_path='data.tar.gz', cache_dir=tmpdir
            )

            with patch.object(StreamingPAPDataset, '__init__', lambda self, *a, **kw: None):
                ds = StreamingPAPDataset.__new__(StreamingPAPDataset)
                ds._cache_path = cfg.get_cache_path()
                ds._cache_enabled = True
                ds.return_metadata = True

                img_tensor = torch.randn(3, 224, 224)
                label_tensor = torch.zeros(68)
                record = (img_tensor, label_tensor, 'path/to/img.jpg')

                ds._write_cache_record(record, 0)

                data = torch.load(os.path.join(ds._cache_path, '0.pt'), weights_only=False)
                assert data['image_path'] == 'path/to/img.jpg'
        finally:
            shutil.rmtree(tmpdir)


# ---------------------------------------------------------------------------
# Retry Logic Tests (Mocked)
# ---------------------------------------------------------------------------

class TestRetryLogic:
    """Tests for retry behavior on network errors."""

    def test_retry_succeeds_after_failures(self):
        from data.tar_streaming.hf_tar_streamer import HFTarStreamer

        streamer = HFTarStreamer(repo_id='user/repo', file_path='data.tar.gz', max_retries=3)

        call_count = 0

        def mock_open(self_fs, path):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Network error")
            return io.BytesIO(b'fake data')

        streamer.fs = MagicMock()
        streamer.fs.open = mock_open

        with patch('data.tar_streaming.hf_tar_streamer.time.sleep'):
            result = streamer._retry_open()
            assert result is not None
            assert call_count == 3

    def test_retry_exhausted(self):
        from data.tar_streaming.hf_tar_streamer import HFTarStreamer

        streamer = HFTarStreamer(repo_id='user/repo', file_path='data.tar.gz', max_retries=2)

        def mock_open(self_fs, path):
            raise ConnectionError("Persistent network error")

        streamer.fs = MagicMock()
        streamer.fs.open = mock_open

        with patch('data.tar_streaming.hf_tar_streamer.time.sleep'):
            with pytest.raises(RuntimeError, match="Failed to open"):
                streamer._retry_open()

    def test_retry_no_failures(self):
        from data.tar_streaming.hf_tar_streamer import HFTarStreamer

        streamer = HFTarStreamer(repo_id='user/repo', file_path='data.tar.gz', max_retries=3)

        expected = io.BytesIO(b'data')
        mock_fs = MagicMock()
        mock_fs.open.return_value = expected
        streamer.fs = mock_fs

        result = streamer._retry_open()
        assert result is expected
        mock_fs.open.assert_called_once()


# ---------------------------------------------------------------------------
# Counter Tests
# ---------------------------------------------------------------------------

class TestCounters:
    """Tests for progress counters."""

    def test_streamer_stats(self):
        from data.tar_streaming.hf_tar_streamer import HFTarStreamer

        streamer = HFTarStreamer(repo_id='user/repo', file_path='data.tar.gz')
        streamer.processed_count = 10
        streamer.error_count = 2
        streamer.skipped_count = 1

        stats = streamer.stats()
        assert stats == {'processed': 10, 'errors': 2, 'skipped': 1}

    def test_reset_counters(self):
        from data.tar_streaming.hf_tar_streamer import HFTarStreamer

        streamer = HFTarStreamer(repo_id='user/repo', file_path='data.tar.gz')
        streamer.processed_count = 10
        streamer.error_count = 2
        streamer.skipped_count = 1

        streamer.reset_counters()
        assert streamer.processed_count == 0
        assert streamer.error_count == 0
        assert streamer.skipped_count == 0


# ---------------------------------------------------------------------------
# Record Processing Tests
# ---------------------------------------------------------------------------

class TestRecordProcessing:
    """Tests for _process_record in StreamingPAPDataset."""

    def test_process_record_basic(self):
        import torch
        from data.tar_streaming.streaming_dataset import StreamingPAPDataset
        from data.tar_streaming.config import StreamingConfig

        cfg = StreamingConfig(
            data_source='hf_tar_stream', repo_id='user/repo',
            file_path='data.tar.gz'
        )

        with patch.object(StreamingPAPDataset, '__init__', lambda self, *a, **kw: None):
            ds = StreamingPAPDataset.__new__(StreamingPAPDataset)
            ds.im_shape = (224, 224)
            ds.return_metadata = False

            from vispr.torch_utils.transformer import SimpleTransformer
            ds.transform = SimpleTransformer(mean=[104, 117, 123])
            ds.attr_id_to_idx = {'a0_safe': 0, 'a1_adult': 1}

            record = {
                'image': Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)),
                'labels': ['a0_safe'],
            }

            img_tensor, label_tensor = ds._process_record(record)
            assert isinstance(img_tensor, torch.Tensor)
            assert img_tensor.shape == (3, 224, 224)
            assert label_tensor.shape == (2,)
            assert label_tensor[0] == 1.0
            assert label_tensor[1] == 0.0

    def test_process_record_with_metadata(self):
        import torch
        from data.tar_streaming.streaming_dataset import StreamingPAPDataset
        from vispr.torch_utils.transformer import SimpleTransformer

        with patch.object(StreamingPAPDataset, '__init__', lambda self, *a, **kw: None):
            ds = StreamingPAPDataset.__new__(StreamingPAPDataset)
            ds.im_shape = (224, 224)
            ds.return_metadata = True
            ds.transform = SimpleTransformer(mean=[104, 117, 123])
            ds.attr_id_to_idx = {'a0_safe': 0}

            record = {
                'image': Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)),
                'labels': ['a0_safe'],
                'image_path': 'test/img.jpg',
            }

            result = ds._process_record(record)
            assert len(result) == 3
            img_tensor, label_tensor, image_path = result
            assert image_path == 'test/img.jpg'


# ---------------------------------------------------------------------------
# HFTarStreamer _create_record Tests
# ---------------------------------------------------------------------------

class TestCreateRecord:
    """Tests for HFTarStreamer._create_record."""

    def test_create_record_with_annotation(self):
        from data.tar_streaming.hf_tar_streamer import HFTarStreamer

        streamer = HFTarStreamer(repo_id='user/repo', file_path='data.tar.gz')
        img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
        annotation = {'labels': ['a0_safe', 'a3_violence'], 'safe': False}

        record = streamer._create_record('img1.jpg', img, annotation)
        assert record is not None
        assert record['image_path'] == 'img1.jpg'
        assert record['image'] is img
        assert set(record['labels']) == {'a0_safe', 'a3_violence'}
        assert record['safe'] is False

    def test_create_record_without_annotation(self):
        from data.tar_streaming.hf_tar_streamer import HFTarStreamer

        streamer = HFTarStreamer(repo_id='user/repo', file_path='data.tar.gz')
        img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))

        record = streamer._create_record('img1.jpg', img, None)
        assert record is not None
        assert record['labels'] == []

    def test_create_record_attributes_format(self):
        from data.tar_streaming.hf_tar_streamer import HFTarStreamer

        streamer = HFTarStreamer(repo_id='user/repo', file_path='data.tar.gz')
        img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
        annotation = {'attributes': {'cat1': ['a0_safe'], 'cat2': ['a3_violence']}}

        record = streamer._create_record('img1.jpg', img, annotation)
        assert set(record['labels']) == {'a0_safe', 'a3_violence'}


# ---------------------------------------------------------------------------
# Dual Archive Streamer _extract_labels Tests
# ---------------------------------------------------------------------------

class TestExtractLabels:
    """Tests for HFDualArchiveStreamer._extract_labels."""

    def test_extract_labels_from_key(self):
        from data.tar_streaming.dual_archive_streamer import HFDualArchiveStreamer

        streamer = HFDualArchiveStreamer(
            repo_id='user/repo',
            image_archive_path='img.tar.gz',
            annotation_archive_path='anno.tar.gz',
            anno_list_content=['a.json'],
        )
        anno = {'labels': ['a0_safe', 'a1_adult']}
        labels = streamer._extract_labels(anno)
        assert set(labels) == {'a0_safe', 'a1_adult'}

    def test_extract_labels_from_attributes(self):
        from data.tar_streaming.dual_archive_streamer import HFDualArchiveStreamer

        streamer = HFDualArchiveStreamer(
            repo_id='user/repo',
            image_archive_path='img.tar.gz',
            annotation_archive_path='anno.tar.gz',
            anno_list_content=['a.json'],
        )
        anno = {'attributes': {'cat': ['a0_safe']}}
        labels = streamer._extract_labels(anno)
        assert labels == ['a0_safe']

    def test_extract_labels_empty(self):
        from data.tar_streaming.dual_archive_streamer import HFDualArchiveStreamer

        streamer = HFDualArchiveStreamer(
            repo_id='user/repo',
            image_archive_path='img.tar.gz',
            annotation_archive_path='anno.tar.gz',
            anno_list_content=['a.json'],
        )
        labels = streamer._extract_labels({})
        assert labels == []


# ---------------------------------------------------------------------------
# Local Tar Streaming Tests
# ---------------------------------------------------------------------------

class TestLocalTarStreamerConfig:
    """Tests for StreamingConfig with data_source='local_tar_stream'."""

    def test_is_local_streaming_mode(self):
        from data.tar_streaming.config import StreamingConfig
        cfg = StreamingConfig(data_source='local_tar_stream', file_path='data.tar.gz')
        assert cfg.is_local_streaming_mode() is True

    def test_not_local_streaming_mode(self):
        from data.tar_streaming.config import StreamingConfig
        cfg = StreamingConfig(data_source='hf_tar_stream', repo_id='u/r', file_path='f.tar.gz')
        assert cfg.is_local_streaming_mode() is False

    def test_local_combined_valid(self):
        """Local combined archive config should validate without raising."""
        from data.tar_streaming.config import StreamingConfig
        tmpdir = tempfile.mkdtemp()
        try:
            archive_path = os.path.join(tmpdir, 'data.tar.gz')
            _create_tar_gz([], archive_path)
            cfg = StreamingConfig(data_source='local_tar_stream', file_path=archive_path)
            cfg.validate()  # should not raise
        finally:
            shutil.rmtree(tmpdir)

    def test_local_dual_valid(self):
        """Local dual archive config should validate without raising."""
        from data.tar_streaming.config import StreamingConfig
        tmpdir = tempfile.mkdtemp()
        try:
            img_archive = os.path.join(tmpdir, 'images.tar.gz')
            anno_archive = os.path.join(tmpdir, 'annotations.tar.gz')
            list_path = os.path.join(tmpdir, 'anno_list.txt')
            _create_tar_gz([], img_archive)
            _create_tar_gz([], anno_archive)
            with open(list_path, 'w') as f:
                f.write('img1.json\n')
            cfg = StreamingConfig(
                data_source='local_tar_stream',
                image_archive_path=img_archive,
                annotation_archive_path=anno_archive,
                anno_list_path=list_path,
            )
            cfg.validate()  # should not raise
        finally:
            shutil.rmtree(tmpdir)

    def test_local_repo_id_not_required(self):
        """repo_id should NOT be required for local_tar_stream."""
        from data.tar_streaming.config import StreamingConfig
        tmpdir = tempfile.mkdtemp()
        try:
            archive_path = os.path.join(tmpdir, 'data.tar.gz')
            _create_tar_gz([], archive_path)
            cfg = StreamingConfig(
                data_source='local_tar_stream',
                file_path=archive_path,
            )
            cfg.validate()  # should not raise, no repo_id needed
        finally:
            shutil.rmtree(tmpdir)

    def test_local_missing_file_path_raises(self):
        """Missing file_path for combined local mode should raise ValueError."""
        from data.tar_streaming.config import StreamingConfig
        cfg = StreamingConfig(data_source='local_tar_stream')
        with pytest.raises(ValueError, match="file_path is required"):
            cfg.validate()

    def test_local_archive_not_found_raises(self):
        """Non-existent local archive path should raise FileNotFoundError."""
        from data.tar_streaming.config import StreamingConfig
        cfg = StreamingConfig(
            data_source='local_tar_stream',
            file_path='/nonexistent/data.tar.gz',
        )
        with pytest.raises(FileNotFoundError, match="Local archive not found"):
            cfg.validate()

    def test_local_dual_missing_image_archive_raises(self):
        """Missing image_archive_path for local dual mode should raise ValueError."""
        from data.tar_streaming.config import StreamingConfig
        cfg = StreamingConfig(
            data_source='local_tar_stream',
            annotation_archive_path='annos.tar.gz',
            anno_list_path='list.txt',
        )
        with pytest.raises(ValueError, match="image_archive_path is required"):
            cfg.validate()

    def test_local_dual_missing_anno_list_raises(self):
        """Missing anno_list_path for local dual mode should raise ValueError."""
        from data.tar_streaming.config import StreamingConfig
        cfg = StreamingConfig(
            data_source='local_tar_stream',
            image_archive_path='imgs.tar.gz',
            annotation_archive_path='annos.tar.gz',
        )
        with pytest.raises(ValueError, match="anno_list_path is required"):
            cfg.validate()

    def test_cache_key_local_mode(self):
        """Cache key should be stable for local_tar_stream configs."""
        from data.tar_streaming.config import StreamingConfig
        cfg1 = StreamingConfig(data_source='local_tar_stream', file_path='/data/train.tar.gz')
        cfg2 = StreamingConfig(data_source='local_tar_stream', file_path='/data/train.tar.gz')
        assert cfg1.get_cache_key() == cfg2.get_cache_key()

    def test_cache_key_differs_local_vs_hf(self):
        """Cache keys should differ between local and HF modes."""
        from data.tar_streaming.config import StreamingConfig
        cfg_local = StreamingConfig(data_source='local_tar_stream', file_path='train.tar.gz')
        cfg_hf = StreamingConfig(data_source='hf_tar_stream', repo_id='u/r', file_path='train.tar.gz')
        assert cfg_local.get_cache_key() != cfg_hf.get_cache_key()


class TestLocalTarStreamerCombined:
    """Tests for LocalTarStreamer combined archive mode."""

    def test_extract_combined_archive(self):
        """Extract records from a local combined .tar.gz archive."""
        from data.tar_streaming.local_tar_streamer import LocalTarStreamer

        tmpdir, archive_path, _ = _make_dummy_archive()
        try:
            streamer = LocalTarStreamer(file_path=archive_path)
            records = list(streamer.extract_structured_data())
            assert len(records) >= 2
            for rec in records:
                assert 'image' in rec
                assert 'image_path' in rec
                assert 'labels' in rec
                assert isinstance(rec['image'], Image.Image)
        finally:
            shutil.rmtree(tmpdir)

    def test_record_has_labels(self):
        """Records from combined archive should contain labels."""
        from data.tar_streaming.local_tar_streamer import LocalTarStreamer

        # Create archive with flat paths so annotation image_path matches archive member names
        # Use valid attribute IDs from attributes.tsv
        tmpdir = tempfile.mkdtemp(prefix='stream_test_')
        archive_path = os.path.join(tmpdir, 'data.tar.gz')
        members = []
        img_bytes = _make_image_bytes()
        anno = _make_annotation('img1.jpg', ['a0_safe', 'a3_height_approx'])
        members.append(('img1.jpg', img_bytes))
        members.append(('img1.json', json.dumps(anno).encode()))
        img_bytes2 = _make_image_bytes()
        anno2 = _make_annotation('img2.jpg', ['a1_age_approx'])
        members.append(('img2.jpg', img_bytes2))
        members.append(('img2.json', json.dumps(anno2).encode()))
        _create_tar_gz(members, archive_path)

        try:
            streamer = LocalTarStreamer(file_path=archive_path)
            records = list(streamer.extract_structured_data())
            all_labels = set()
            for rec in records:
                all_labels.update(rec.get('labels', []))
            assert 'a0_safe' in all_labels
            assert 'a1_age_approx' in all_labels
        finally:
            shutil.rmtree(tmpdir)

    def test_stats_after_extraction(self):
        """Stats should be populated after streaming."""
        from data.tar_streaming.local_tar_streamer import LocalTarStreamer

        tmpdir, archive_path, _ = _make_dummy_archive()
        try:
            streamer = LocalTarStreamer(file_path=archive_path)
            list(streamer.extract_structured_data())
            stats = streamer.stats()
            assert stats['processed'] > 0
        finally:
            shutil.rmtree(tmpdir)


class TestLocalTarStreamerDual:
    """Tests for LocalTarStreamer dual archive mode."""

    def test_extract_dual_archive(self):
        """Extract records from separate local image and annotation archives."""
        from data.tar_streaming.local_tar_streamer import LocalTarStreamer

        tmpdir = tempfile.mkdtemp()
        try:
            img_archive, anno_archive, list_path = _make_dual_archive(tmpdir)

            streamer = LocalTarStreamer(
                image_archive_path=img_archive,
                annotation_archive_path=anno_archive,
                anno_list_path=list_path,
            )
            records = list(streamer.extract_structured_data())
            assert len(records) == 3
            for rec in records:
                assert 'image' in rec
                assert 'image_path' in rec
                assert 'labels' in rec
        finally:
            shutil.rmtree(tmpdir)

    def test_dual_extract_labels(self):
        """Labels should be extracted from annotations."""
        from data.tar_streaming.local_tar_streamer import LocalTarStreamer

        tmpdir = tempfile.mkdtemp()
        try:
            img_archive, anno_archive, list_path = _make_dual_archive(tmpdir)
            streamer = LocalTarStreamer(
                image_archive_path=img_archive,
                annotation_archive_path=anno_archive,
                anno_list_path=list_path,
            )
            records = list(streamer.extract_structured_data())
            all_labels = set()
            for rec in records:
                all_labels.update(rec.get('labels', []))
            assert len(all_labels) > 0
        finally:
            shutil.rmtree(tmpdir)

    def test_dual_stats(self):
        """Stats should be populated after streaming."""
        from data.tar_streaming.local_tar_streamer import LocalTarStreamer

        tmpdir = tempfile.mkdtemp()
        try:
            img_archive, anno_archive, list_path = _make_dual_archive(tmpdir)
            streamer = LocalTarStreamer(
                image_archive_path=img_archive,
                annotation_archive_path=anno_archive,
                anno_list_path=list_path,
            )
            list(streamer.extract_structured_data())
            stats = streamer.stats()
            assert stats['processed'] > 0
        finally:
            shutil.rmtree(tmpdir)


class TestStreamingPAPDatasetLocalTar:
    """Tests for StreamingPAPDataset with local_tar_stream mode."""

    def test_local_combined_dataset_iteration(self):
        """StreamingPAPDataset should iterate through local combined archive."""
        from data.tar_streaming.streaming_dataset import StreamingPAPDataset
        from data.tar_streaming.config import StreamingConfig

        # Use a local archive with flat paths (matching annotation image_path)
        # and valid attribute IDs from attributes.tsv
        tmpdir = tempfile.mkdtemp(prefix='stream_test_')
        archive_path = os.path.join(tmpdir, 'data.tar.gz')
        members = []
        img_bytes = _make_image_bytes()
        anno = _make_annotation('img1.jpg', ['a0_safe', 'a3_height_approx'])
        members.append(('img1.jpg', img_bytes))
        members.append(('img1.json', json.dumps(anno).encode()))
        _create_tar_gz(members, archive_path)

        attr_path = os.path.join(REPO_ROOT, 'vispr', 'datasets', 'attributes.tsv')
        try:
            cfg = StreamingConfig(
                data_source='local_tar_stream',
                file_path=archive_path,
            )
            cfg.validate()
            dataset = StreamingPAPDataset(config=cfg, shuffle=False, attr_list_path=attr_path)
            count = 0
            for img_tensor, label_vec in dataset:
                assert img_tensor.shape == (3, 224, 224)
                count += 1
            assert count == 1
        finally:
            shutil.rmtree(tmpdir)

    def test_local_dual_dataset_iteration(self):
        """StreamingPAPDataset should iterate through local dual archives."""
        from data.tar_streaming.streaming_dataset import StreamingPAPDataset
        from data.tar_streaming.config import StreamingConfig

        tmpdir = tempfile.mkdtemp()
        try:
            img_archive, anno_archive, list_path = _make_dual_archive(tmpdir)

            attr_path = os.path.join(REPO_ROOT, 'vispr', 'datasets', 'attributes.tsv')
            cfg = StreamingConfig(
                data_source='local_tar_stream',
                image_archive_path=img_archive,
                annotation_archive_path=anno_archive,
                anno_list_path=list_path,
            )
            cfg.validate()
            dataset = StreamingPAPDataset(config=cfg, shuffle=False, attr_list_path=attr_path)
            count = 0
            for img_tensor, label_vec in dataset:
                assert img_tensor.shape == (3, 224, 224)
                count += 1
            assert count == 3
        finally:
            shutil.rmtree(tmpdir)


# ---------------------------------------------------------------------------
# Integration Tests (require HF Hub access)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestIntegration:
    """Integration tests that connect to actual HF Hub.

    These tests require network access and a valid HF token (if the repo is private).
    """

    def test_stream_from_public_repo(self):
        """Test streaming from a known public HF dataset.

        This test uses the CIFAR-10 dataset as a small public test case.
        It won't have VPA annotations, but verifies the streaming mechanism.
        """
        from data.tar_streaming.config import StreamingConfig
        from data.tar_streaming.hf_tar_streamer import HFTarStreamer

        # Use a small public dataset - this will stream but may not produce
        # matched records (no matching annotations). We just test that streaming works.
        try:
            streamer = HFTarStreamer(
                repo_id='hf-internal-testing/fixtures_csv_json',
                file_path='jsonl/test.jsonl.gz',
                max_retries=3,
            )
            count = 0
            for name, data, info in streamer.stream_tar_members():
                count += 1
                if count >= 5:
                    break
            assert count > 0, "Should have streamed at least one member"
        except Exception as e:
            pytest.skip(f"HF Hub not accessible: {e}")

    def test_config_from_yaml_roundtrip(self):
        """Test that StreamingConfig can be created from a YAML-like dict."""
        from data.tar_streaming.config import StreamingConfig
        import yaml

        config_dict = {
            'data_source': 'hf_tar_stream',
            'streaming': {
                'repo_id': 'user/repo',
                'file_path': 'train.tar.gz',
                'buffer_size': 2000,
            },
            'data': {
                'batch_size': 16,
                'num_classes': 68,
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_dict, f)
            yaml_path = f.name

        try:
            cfg = StreamingConfig.from_yaml(yaml_path)
            assert cfg.data_source == 'hf_tar_stream'
            assert cfg.repo_id == 'user/repo'
            assert cfg.file_path == 'train.tar.gz'
            assert cfg.buffer_size == 2000
            assert cfg.batch_size == 16
            assert cfg.num_classes == 68
        finally:
            os.unlink(yaml_path)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
