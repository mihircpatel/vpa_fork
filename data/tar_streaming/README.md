# HuggingFace Tar Streaming Module

Stream `.tar.gz` datasets directly from Hugging Face Hub without downloading to local disk. Perfect for disk-constrained environments like Google Colab.

## Features

- **Zero Disk Usage**: Stream archives directly from HF Hub using range requests
- **Memory-Efficient**: In-memory tar extraction with configurable shuffle buffer
- **Drop-in Replacement**: Compatible with existing `PAPDataset` interface
- **Config-Driven**: Toggle between local and streaming modes via YAML or CLI
- **Nested Directory Support**: Automatically handles complex archive structures
- **Local Caching**: Cache processed records locally to avoid re-streaming
- **Retry with Backoff**: Automatic retry on network errors with exponential backoff
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
├── hf_tar_streamer.py        # Combined archive streamer (retry, validation)
├── dual_archive_streamer.py  # Dual archive streamer (retry, validation)
├── streaming_dataset.py      # PyTorch Dataset adapter (cache, progress logging)
└── example_usage.py          # Runnable usage examples
```

### Key Components

1. **`StreamingConfig`**: Configuration dataclass with validation
   - Loads from YAML, CLI args, or direct construction
   - Supports local and streaming modes
   - Caching, retry, and logging configuration

2. **`HFTarStreamer`**: Combined archive streamer
   - Streams `.tar.gz` from HF Hub using `HfFileSystem`
   - Retry with exponential backoff on network errors
   - Integrity validation (empty data, image header checks)
   - Progress counters: `processed`, `errors`, `skipped`

3. **`HFDualArchiveStreamer`**: Dual archive streamer
   - Same retry/validation as `HFTarStreamer`
   - Matches images with annotations across separate archives

4. **`StreamingPAPDataset`**: PyTorch `IterableDataset`
   - Compatible with existing `PAPDataset` interface
   - Shuffle buffer for randomization
   - Optional local caching (record-level, no invalidation)
   - Progress logging at configurable intervals

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_source` | str | `"local"` | `"local"` or `"hf_tar_stream"` |
| `repo_id` | str | None | HuggingFace repo (e.g., `"username/dataset"`) |
| `file_path` | str | None | Path to `.tar.gz` in repo |
| `buffer_size` | int | 1000 | Shuffle buffer size (higher = better shuffle) |
| `im_shape` | tuple | (224, 224) | Image shape (height, width) |
| `mean` | tuple | (104, 117, 123) | Mean for normalization [B, G, R] |
| `num_classes` | int | 68 | Number of attribute classes |
| `batch_size` | int | 32 | Batch size |
| `chunk_size` | int | 8388608 | Read chunk size in bytes (8MB) |
| `log_interval` | int | 100 | Log progress every N records (0=disabled) |
| `max_retries` | int | 3 | Max retries on network errors |
| `cache_dir` | str | None | Local cache directory (None=disabled) |

## Comparison: Local vs Streaming

| Aspect | Local Loading | HF Streaming |
|--------|---------------|--------------|
| Disk Usage | Full dataset | ~0 MB (or cache size) |
| Memory Usage | Metadata only | Shuffle buffer only |
| Setup Time | Download required | Instant |
| Random Access | Yes | Sequential (buffered) |
| Speed | Faster (local SSD) | Network-dependent |
| Use Case | Development | Colab, limited disk |

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

## Backward Compatibility

The module is **100% backward compatible**:

- Existing scripts work unchanged with `--infile` argument
- Default `data_source` is `"local"`
- No changes to existing `PAPDataset` or training logic
- Streaming is opt-in via command-line flag or config
- All new parameters have sensible defaults

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
