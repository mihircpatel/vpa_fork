# HuggingFace Tar Streaming Module

Stream `.tar.gz` datasets directly from Hugging Face Hub without downloading to local disk. Perfect for disk-constrained environments like Google Colab.

## Features

- **Zero Disk Usage**: Stream archives directly from HF Hub using range requests
- **Memory-Efficient**: In-memory tar extraction with configurable shuffle buffer
- **Drop-in Replacement**: Compatible with existing `PAPDataset` interface
- **Config-Driven**: Toggle between local and streaming modes via YAML or CLI
- **Nested Directory Support**: Automatically handles complex archive structures

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

# Create config
config = StreamingConfig(
    data_source='hf_tar_stream',
    repo_id='username/dataset-name',
    file_path='train.tar.gz',
    buffer_size=1000,
    batch_size=32
)

# Create dataset
dataset = StreamingPAPDataset(config=config, shuffle=True)

# Create data loader
loader = DataLoader(dataset, batch_size=32, num_workers=0)

# Train as usual
for images, labels in loader:
    # Your training loop
    pass
```

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
├── __init__.py              # Module exports
├── config.py                # Configuration management
├── hf_tar_streamer.py       # Core streaming logic
└── streaming_dataset.py     # PyTorch Dataset adapter
```

### Key Components

1. **`HFTarStreamer`**: Streams `.tar.gz` from HF Hub using `HfFileSystem`
   - Opens remote file stream without downloading
   - Extracts members in-memory using `tarfile`
   - Matches images with annotations

2. **`StreamingPAPDataset`**: PyTorch `IterableDataset`
   - Compatible with existing `PAPDataset` interface
   - Implements shuffle buffer for randomization
   - Applies same transformations as local dataset

3. **`StreamingConfig`**: Configuration dataclass
   - Loads from YAML or command-line args
   - Validates configuration
   - Supports both local and streaming modes

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

## Comparison: Local vs Streaming

| Aspect | Local Loading | HF Streaming |
|--------|---------------|--------------|
| Disk Usage | Full dataset | ~0 MB |
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

4. **Network**:
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

### Slow streaming

- Check internet connection
- Try smaller batch size initially
- Increase `buffer_size` if enough memory

### Out of Memory (OOM)

- Reduce `buffer_size` (e.g., from 2000 to 500)
- Reduce `batch_size`
- Close other applications

## Examples

### Google Colab Example

```python
# Install dependencies
!pip install huggingface_hub PyYAML

# Clone repo (if needed)
!git clone https://github.com/your-repo/vpa_fork.git
%cd vpa_fork

# Stream and train
!python vispr/tools/scripts/train_torch.py \
    --data-source hf_tar_stream \
    --hf-repo username/vispr-dataset \
    --hf-file-path train.tar.gz \
    --arch resnet50 \
    --pretrained \
    --epochs 5 \
    --batch-size 16 \
    --buffer-size 1000
```

### Hybrid Approach (Local Val, Streaming Train)

```bash
# Train on streaming data, validate on local
python vispr/tools/scripts/train_torch.py \
    --data-source hf_tar_stream \
    --hf-repo username/dataset \
    --hf-file-path huge_train.tar.gz \
    --valfile val2017.txt \
    --arch resnet50 \
    --epochs 10
```

## Backward Compatibility

The module is **100% backward compatible**:

- Existing scripts work unchanged with `--infile` argument
- Default `data_source` is `"local"`
- No changes to existing `PAPDataset` or training logic
- Streaming is opt-in via command-line flag or config

## Performance Notes

- **First Epoch**: Slightly slower as data streams over network
- **Subsequent Epochs**: Data re-streams (no caching to save disk)
- **Throughput**: Network-bound, typically 50-200 MB/s on HF Hub
- **Latency**: Minimal impact due to prefetching in buffer

## Contributing

The streaming module is self-contained in `data/tar_streaming/`. To extend:

1. **New Streaming Sources**: Subclass `HFTarStreamer`
2. **Custom Transforms**: Pass custom `transform` to `StreamingPAPDataset`
3. **New Archive Formats**: Modify `_is_image_file()` and extraction logic

## License

Same as parent project. See LICENSE in repository root.
