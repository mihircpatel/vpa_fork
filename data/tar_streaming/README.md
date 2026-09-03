# Tar Streaming Module

Stream `.tar.gz` datasets from Hugging Face Hub **or local disk** without extracting them to a directory tree. Perfect for disk-constrained environments like Google Colab, or when you want to train directly from a tar archive without inflating it on disk.

## Features

- **Zero Disk Usage**: Stream archives directly from HF Hub using range requests
- **Memory-Efficient**: In-memory tar extraction with configurable shuffle buffer
- **Drop-in Replacement**: Compatible with existing `PAPDataset` interface
- **Config-Driven**: Toggle between local, HF streaming, and local tar streaming via YAML or CLI
- **Nested Directory Support**: Automatically handles complex archive structures
- **Local Caching**: Cache processed records locally to avoid re-streaming
- **Retry with Backoff**: Automatic retry on network errors with exponential backoff (HF mode)
- **Integrity Validation**: Detects corrupted or truncated files before processing
- **Progress Logging**: Configurable logging of streaming progress and counters

## Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```

Or install streaming dependencies separately:

```bash
pip install huggingface_hub PyYAML
```

## Quick Start

### Option 1: Command-Line Arguments

Stream training data from HuggingFace:

```bash
python vispr/tools/scripts/train_torch.py \
    --data-source hf_tar_stream \
    --hf-repo username/dataset-name \
    --hf-file-path train_data.tar.gz \
    --hf-val-file-path val_data.tar.gz \
    --buffer-size 2000 \
    --arch resnet50 \
    --pretrained \
    --epochs 10 \
    --batch-size 32
```

### Option 2: YAML Configuration

1. Edit `configs/data_config.yaml`:

```yaml
data_source: "hf_tar_stream"

streaming:
  repo_id: "username/vispr-dataset"
  file_path: "train.tar.gz"
  buffer_size: 2000

data:
  batch_size: 32
  num_classes: 68
```

2. Run training:

```bash
python vispr/tools/scripts/train_torch.py \
    --config configs/data_config.yaml \
    --arch resnet50 \
    --pretrained \
    --epochs 10
```

### Option 3: Direct Python API

```python
from data.tar_streaming import StreamingConfig, StreamingPAPDataset
from torch.utils.data import DataLoader

config = StreamingConfig(
    data_source='hf_tar_stream',
    repo_id='username/dataset-name',
    file_path='train.tar.gz',
    buffer_size=1000,
    batch_size=32
)

dataset = StreamingPAPDataset(config=config, shuffle=True)
loader = DataLoader(dataset, batch_size=32, num_workers=0)

for images, labels in loader:
    # Your training loop
    pass
```

### Option 4: Local Tar Streaming (Combined Archive)

Stream from a local `.tar.gz` file on the same machine, without extracting it first:

```bash
python vispr/tools/scripts/train_torch.py \
    --data-source local_tar_stream \
    --local-file-path ./datasets/train_data.tar.gz \
    --local-val-file-path ./datasets/val_data.tar.gz \
    --buffer-size 2000 \
    --arch resnet50 \
    --pretrained \
    --epochs 10 \
    --batch-size 32
```

Or via YAML (`configs/data_config.yaml`):

```yaml
data_source: "local_tar_stream"

streaming:
  file_path: "./datasets/train_data.tar.gz"
  buffer_size: 2000

data:
  batch_size: 32
  num_classes: 68
```

Or via Python API:

```python
from data.tar_streaming import StreamingConfig, StreamingPAPDataset
from torch.utils.data import DataLoader

config = StreamingConfig(
    data_source='local_tar_stream',
    file_path='./datasets/train_data.tar.gz',
    buffer_size=1000,
    batch_size=32
)

dataset = StreamingPAPDataset(config=config, shuffle=True)
loader = DataLoader(dataset, batch_size=32, num_workers=0)

for images, labels in loader:
    # Your training loop
    pass
```

### Option 5: Local Tar Streaming (Dual Archive)

Use separate local image and annotation archives:

```bash
python vispr/tools/scripts/train_torch.py \
    --data-source local_tar_stream \
    --local-image-archive ./datasets/train_images.tar.gz \
    --local-anno-archive ./datasets/train_annotations.tar.gz \
    --local-anno-list ./datasets/train_anno_list.txt \
    --local-val-image-archive ./datasets/val_images.tar.gz \
    --local-val-anno-archive ./datasets/val_annotations.tar.gz \
    --local-val-anno-list ./datasets/val_anno_list.txt \
    --buffer-size 2000 \
    --arch resnet50 \
    --pretrained \
    --epochs 10
```

## Local Caching

Cache processed records locally to avoid re-streaming on subsequent runs:

```python
config = StreamingConfig(
    data_source='hf_tar_stream',
    repo_id='username/dataset',
    file_path='train.tar.gz',
    cache_dir='./stream_cache',    # Enable caching
    max_retries=5,                 # Retry on network errors
    log_interval=50,               # Log every 50 records
)

dataset = StreamingPAPDataset(config=config, shuffle=True)
# First run: streams from HF Hub, writes cache
# Second run: reads from cache (no network needed)
```

Or via CLI:

```bash
python vispr/tools/scripts/train_torch.py \
    --data-source hf_tar_stream \
    --hf-repo username/dataset \
    --hf-file-path train.tar.gz \
    --cache-dir ./stream_cache \
    --max-retries 5
```

To invalidate cache, delete the cache directory:
```bash
rm -rf ./stream_cache
```

## Error Handling and Retry

The streamer automatically retries on network errors with exponential backoff:

```python
config = StreamingConfig(
    data_source='hf_tar_stream',
    repo_id='username/dataset',
    file_path='train.tar.gz',
    max_retries=5,      # Retry up to 5 times (backoff: 2s, 4s, 8s, 16s, 30s)
    log_interval=10,    # Log progress frequently for debugging
)

dataset = StreamingPAPDataset(config=config, shuffle=True)
# After iteration, check stats:
stats = dataset.stats()
print(f"Processed: {stats['streamer']['processed']}")
print(f"Errors: {stats['streamer']['errors']}")
print(f"Skipped: {stats['streamer']['skipped']}")
```

**What gets retried:**
- Network connection failures
- Timeout errors
- Transient HTTP errors from HuggingFace Hub

**What gets skipped (not retried):**
- Corrupted image files (detected via header validation)
- Malformed JSON annotations
- Empty tar members

## Archive Structure Requirements

Your `.tar.gz` archive should contain:

1. **Images**: `.jpg`, `.jpeg`, `.png`, etc.
2. **Annotations** (optional): `.json` files with structure:

```json
{
  "image_path": "path/to/image.jpg",
  "labels": ["a0_safe", "a3_violence"],
  "safe": false,
  "attributes": {
    "category1": ["attr1", "attr2"]
  }
}
```

The module handles:
- Nested directories (flattens automatically)
- Images with or without annotations
- Multiple annotation formats

## Architecture

```
data/tar_streaming/
├── __init__.py               # Module exports
├── config.py                 # Configuration dataclass
├── hf_tar_streamer.py        # HF combined archive streamer (retry, validation)
├── dual_archive_streamer.py  # HF dual archive streamer (retry, validation)
├── local_tar_streamer.py     # Local tar.gz streamer (combined & dual modes)
├── streaming_dataset.py      # PyTorch Dataset adapter (cache, progress logging)
└── example_usage.py          # Runnable usage examples
```

### Key Components

1. **`StreamingConfig`**: Configuration dataclass with validation
   - Loads from YAML, CLI args, or direct construction
   - Supports `local`, `hf_tar_stream` and `local_tar_stream` data sources
   - Caching, retry, and logging configuration

2. **`HFTarStreamer`**: HF combined archive streamer
   - Streams `.tar.gz` from HF Hub using `HfFileSystem`
   - Retry with exponential backoff on network errors
   - Integrity validation (empty data, image header checks)
   - Progress counters: `processed`, `errors`, `skipped`

3. **`HFDualArchiveStreamer`**: HF dual archive streamer
   - Same retry/validation as `HFTarStreamer`
   - Matches images with annotations across separate archives

4. **`LocalTarStreamer`**: Local tar.gz streamer
   - Streams `.tar.gz` archives from local disk
   - Supports both combined and dual archive modes
   - No network retry needed (local I/O)
   - Same record interface as HF streamers

5. **`StreamingPAPDataset`**: PyTorch `IterableDataset`
   - Compatible with existing `PAPDataset` interface
   - Automatically selects the correct streamer based on `data_source`
   - Shuffle buffer for randomization
   - Optional local caching (record-level, no invalidation)
   - Progress logging at configurable intervals

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_source` | str | `"local"` | `"local"`, `"hf_tar_stream"` or `"local_tar_stream"` |
| `repo_id` | str | None | HuggingFace repo (e.g., `"username/dataset"`). Required for `hf_tar_stream` only. |
| `file_path` | str | None | Path to `.tar.gz` (HF path for `hf_tar_stream`, local path for `local_tar_stream`) |
| `image_archive_path` | str | None | Path to image `.tar.gz` (dual archive mode) |
| `annotation_archive_path` | str | None | Path to annotation `.tar.gz` (dual archive mode) |
| `anno_list_path` | str | None | Path to annotation list `.txt` (dual archive mode) |
| `buffer_size` | int | 1000 | Shuffle buffer size (higher = better shuffle) |
| `im_shape` | tuple | (224, 224) | Image shape (height, width) |
| `mean` | tuple | (104, 117, 123) | Mean for normalization [B, G, R] |
| `num_classes` | int | 68 | Number of attribute classes |
| `batch_size` | int | 32 | Batch size |
| `chunk_size` | int | 8388608 | Read chunk size in bytes (8MB) |
| `log_interval` | int | 100 | Log progress every N records (0=disabled) |
| `max_retries` | int | 3 | Max retries on network errors (HF mode only) |
| `cache_dir` | str | None | Local cache directory (None=disabled) |

## Comparison: Local Loading vs HF Streaming vs Local Tar Streaming

| Aspect | Local Loading | HF Streaming | Local Tar Streaming |
|--------|---------------|--------------|---------------------|
| Disk Usage (data) | Full dataset extracted | ~0 MB | No extraction needed |
| Disk Usage (archive) | N/A | N/A | Reads directly from .tar.gz |
| Memory Usage | Metadata only | Shuffle buffer only | Shuffle buffer only |
| Setup Time | Download + extract | Instant | Instant |
| Random Access | Yes | Sequential (buffered) | Sequential (buffered) |
| Speed | Faster (local SSD) | Network-bound | Fast (local I/O, no network) |
| Network Required | Yes (download) | Yes | No |
| Use Case | Development, local training | Colab, limited disk | Training from local archives, no extraction |

## Best Practices

1. **Buffer Size**:
   - Larger buffer = better shuffling but more memory
   - Recommended: 1000-5000 for most datasets
   - Colab: Keep under 2000 to avoid OOM

2. **Batch Size**:
   - Start with 32 and adjust based on GPU memory
   - Streaming has minimal impact on batch size choice

3. **Workers**:
   - Use `num_workers=0` for streaming (IterableDataset limitation)
   - Local datasets can use `num_workers=2-4`

4. **Caching**:
   - Use `cache_dir` when re-running experiments to save time
   - Delete cache dir to force re-stream if data changes
   - Cache uses ~10-50 MB per 1000 records (tensor storage)

5. **Retry**:
   - Default `max_retries=3` works for stable connections
   - Increase to 5+ for unreliable networks
   - Backoff: 2s, 4s, 8s, 16s, 30s (capped)

6. **Network**:
   - Ensure stable internet connection
   - HF Hub uses CDN for fast downloads
   - Range requests minimize bandwidth

## Troubleshooting

### ImportError: No module named 'huggingface_hub'

Install dependencies:
```bash
pip install huggingface_hub PyYAML
```

### ValueError: repo_id is required

Ensure you provide both `--hf-repo` and `--hf-file-path`:
```bash
--data-source hf_tar_stream --hf-repo username/dataset --hf-file-path train.tar.gz
```

### RuntimeError: Failed to open ... after N attempts

Network connection is failing. Try:
1. Check internet connection
2. Increase `--max-retries` (e.g., `--max-retries 10`)
3. Check if the HF repo/file exists and is public

### Slow streaming

- Check internet connection
- Try smaller batch size initially
- Increase `buffer_size` if enough memory
- Enable caching with `--cache-dir` for subsequent runs

### Out of Memory (OOM)

- Reduce `buffer_size` (e.g., from 2000 to 500)
- Reduce `batch_size`
- Close other applications
- Disable caching if disk is also limited

### Corrupted data warnings

The streamer validates file headers. If you see warnings like:
```
Image header mismatch for foo.jpg: extension=.jpg, detected_header=none
```
The file may be corrupted. It will be skipped automatically.

### Cache issues

If cached data seems stale or corrupted:
```bash
rm -rf ./stream_cache
```
The next run will re-stream from HF Hub.

## Examples

### Google Colab Example

```python
# Install dependencies
!pip install huggingface_hub PyYAML

# Clone repo (if needed)
!git clone https://github.com/your-repo/vpa_fork.git
%cd vpa_fork

# Stream and train with caching
!python vispr/tools/scripts/train_torch.py \
    --data-source hf_tar_stream \
    --hf-repo username/vispr-dataset \
    --hf-file-path train.tar.gz \
    --arch resnet50 \
    --pretrained \
    --epochs 5 \
    --batch-size 16 \
    --buffer-size 1000 \
    --cache-dir ./stream_cache
```

### Hybrid Approach (Local Val, Streaming Train)

```bash
python vispr/tools/scripts/train_torch.py \
    --data-source hf_tar_stream \
    --hf-repo username/dataset \
    --hf-file-path huge_train.tar.gz \
    --valfile val2017.txt \
    --arch resnet50 \
    --epochs 10
```

### Low-Bandwidth / Unreliable Network

```bash
python vispr/tools/scripts/train_torch.py \
    --data-source hf_tar_stream \
    --hf-repo username/dataset \
    --hf-file-path train.tar.gz \
    --max-retries 10 \
    --cache-dir ./stream_cache \
    --buffer-size 500
```

### Local Tar Streaming (no network, no extraction)

Train directly from a local `.tar.gz` without extracting it first:

```bash
python vispr/tools/scripts/train_torch.py \
    --data-source local_tar_stream \
    --local-file-path ./datasets/train_data.tar.gz \
    --local-val-file-path ./datasets/val_data.tar.gz \
    --arch resnet50 \
    --pretrained \
    --epochs 10 \
    --batch-size 32
```

Dual archive mode:

```bash
python vispr/tools/scripts/train_torch.py \
    --data-source local_tar_stream \
    --local-image-archive ./datasets/images.tar.gz \
    --local-anno-archive ./datasets/annotations.tar.gz \
    --local-anno-list ./datasets/train.txt \
    --local-val-image-archive ./datasets/val_images.tar.gz \
    --local-val-anno-archive ./datasets/val_annotations.tar.gz \
    --local-val-anno-list ./datasets/val.txt \
    --arch resnet50 \
    --pretrained \
    --epochs 10
```

## Backward Compatibility

The module is **100% backward compatible**:

- Existing scripts work unchanged with `--infile` argument
- Default `data_source` is `"local"`
- No changes to existing `PAPDataset` or training logic
- Streaming (HF and local) is opt-in via command-line flag or config
- All new parameters have sensible defaults
- New `local_tar_stream` mode is purely additive

## Performance Notes

- **First Epoch**: Slightly slower as data streams over network
- **Subsequent Epochs**: With caching, reads from local disk (fast). Without caching, re-streams.
- **Throughput**: Network-bound, typically 50-200 MB/s on HF Hub
- **Latency**: Minimal impact due to prefetching in buffer
- **Cache Storage**: ~10-50 MB per 1000 records (tensor + label storage)

## Contributing

The streaming module is self-contained in `data/tar_streaming/`. To extend:

1. **New Streaming Sources**: Subclass `HFTarStreamer`
2. **Custom Transforms**: Pass custom `transform` to `StreamingPAPDataset`
3. **New Archive Formats**: Modify `_is_image_file()` and extraction logic
4. **Cache Backends**: Override `_iter_from_cache()` and `_write_cache_record()` in `StreamingPAPDataset`

## Testing

Run the unit tests:
```bash
python -m pytest tests/test_streaming.py -v
```

Run integration tests (requires HF Hub access):
```bash
python -m pytest tests/test_streaming.py -v -m integration
```

## License

Same as parent project. See LICENSE in repository root.
