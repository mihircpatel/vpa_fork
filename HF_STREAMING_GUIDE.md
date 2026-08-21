# HuggingFace Tar Streaming Integration Guide

This guide shows how to use the new HuggingFace tar streaming feature to train models without downloading datasets to local disk.

## Overview

The streaming module allows you to:
- Stream `.tar.gz` archives directly from HuggingFace Hub
- Train models without consuming local disk space
- Perfect for Google Colab and other disk-constrained environments
- Zero modification to existing training logic

## Quick Start (3 Steps)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Or specifically:
```bash
pip install huggingface_hub PyYAML
```

### 2. Upload Your Dataset to HuggingFace Hub

Create a `.tar.gz` archive of your dataset:

```bash
# Example: Create tar archive of images and annotations
tar -czf train.tar.gz train_images/ train_annotations/
tar -czf val.tar.gz val_images/ val_annotations/
```

Upload to HuggingFace Hub:
1. Create a dataset repository on https://huggingface.co
2. Upload your `.tar.gz` files
3. Note your `repo_id` (e.g., `username/vispr-dataset`)

### 3. Train with Streaming

**Option A: Command-line**
```bash
python vispr/tools/scripts/train_torch.py \
    --data-source hf_tar_stream \
    --hf-repo username/vispr-dataset \
    --hf-file-path train.tar.gz \
    --hf-val-file-path val.tar.gz \
    --arch resnet50 \
    --pretrained \
    --epochs 10 \
    --batch-size 32 \
    --buffer-size 1000
```

**Option B: Config file**
```bash
# Edit configs/data_config.yaml first
python vispr/tools/scripts/train_torch.py \
    --config configs/data_config.yaml \
    --arch resnet50 \
    --pretrained \
    --epochs 10
```

## Detailed Usage

### Archive Structure

Your `.tar.gz` should contain images and optional JSON annotations:

```
train.tar.gz
├── images/
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
└── annotations/
    ├── img_001.json
    ├── img_002.json
    └── ...
```

Each JSON annotation should have:
```json
{
  "image_path": "images/img_001.jpg",
  "labels": ["a0_safe", "a3_violence"],
  "safe": false
}
```

### Configuration File

Edit `configs/data_config.yaml`:

```yaml
data_source: "hf_tar_stream"

streaming:
  repo_id: "username/vispr-dataset"
  file_path: "train.tar.gz"
  buffer_size: 1000

data:
  im_shape: [224, 224]
  mean: [104.0, 117.0, 123.0]
  num_classes: 68
  batch_size: 32
```

Then run:
```bash
python vispr/tools/scripts/train_torch.py --config configs/data_config.yaml --arch resnet50 --pretrained --epochs 10
```

### Command-Line Arguments

All streaming options:

| Argument | Description | Example |
|----------|-------------|---------|
| `--data-source` | Data source type | `hf_tar_stream` or `local` |
| `--hf-repo` | HF repository ID | `username/dataset-name` |
| `--hf-file-path` | Training tar.gz path | `train.tar.gz` |
| `--hf-val-file-path` | Validation tar.gz path | `val.tar.gz` |
| `--buffer-size` | Shuffle buffer size | `1000` (higher = better shuffle) |
| `--config` | YAML config file | `configs/data_config.yaml` |

### Python API

For custom training scripts:

```python
from data.tar_streaming import StreamingConfig, StreamingPAPDataset
from torch.utils.data import DataLoader

# Create config
config = StreamingConfig(
    data_source='hf_tar_stream',
    repo_id='username/vispr-dataset',
    file_path='train.tar.gz',
    buffer_size=1000,
    batch_size=32
)

# Create dataset
dataset = StreamingPAPDataset(config=config, shuffle=True)

# Create loader
loader = DataLoader(dataset, batch_size=32, num_workers=0)

# Use in training loop
for images, labels in loader:
    # Your training code
    pass
```

## Google Colab Example

Complete Colab notebook example:

```python
# 1. Install dependencies
!pip install huggingface_hub PyYAML torch torchvision

# 2. Clone repository
!git clone https://github.com/your-org/vpa_fork.git
%cd vpa_fork

# 3. Install remaining requirements
!pip install -r requirements.txt

# 4. Train with streaming (no disk space used!)
!python vispr/tools/scripts/train_torch.py \
    --data-source hf_tar_stream \
    --hf-repo username/vispr-dataset \
    --hf-file-path train.tar.gz \
    --hf-val-file-path val.tar.gz \
    --arch resnet50 \
    --pretrained \
    --epochs 5 \
    --batch-size 16 \
    --buffer-size 1000 \
    --save-path ./checkpoints/model.pth

# 5. Check results
!ls -lh ./checkpoints/
```

## Performance Tuning

### Buffer Size

- **Small (500-1000)**: Low memory, less shuffling
- **Medium (1000-2000)**: Balanced (recommended)
- **Large (2000-5000)**: Better shuffling, more memory

```bash
--buffer-size 2000  # Adjust based on available RAM
```

### Batch Size

Start with smaller batches and increase:

```bash
--batch-size 16  # Safe for most GPUs
--batch-size 32  # Standard
--batch-size 64  # If you have GPU memory
```

### Network Optimization

Streaming performance depends on network speed:
- HuggingFace Hub uses CDN (typically 50-200 MB/s)
- First epoch may be slower as data streams
- Subsequent epochs re-stream (no local caching)

## Troubleshooting

### Error: "Module not found: huggingface_hub"

**Solution:**
```bash
pip install huggingface_hub PyYAML
```

### Error: "repo_id is required"

**Solution:** Provide both `--hf-repo` and `--hf-file-path`:
```bash
--data-source hf_tar_stream --hf-repo username/dataset --hf-file-path train.tar.gz
```

### Error: "Out of Memory"

**Solutions:**
1. Reduce buffer size: `--buffer-size 500`
2. Reduce batch size: `--batch-size 16`
3. Use smaller images in your archive

### Slow streaming

**Solutions:**
1. Check internet connection
2. Try HF Hub mirror if available
3. Use smaller archive files (split large datasets)

### Data not shuffling well

**Solution:** Increase buffer size:
```bash
--buffer-size 3000  # Larger buffer = better shuffling
```

## Migration from Local to Streaming

### Before (Local Loading)

```bash
python vispr/tools/scripts/train_torch.py \
    --infile train2017.txt \
    --valfile val2017.txt \
    --arch resnet50 \
    --epochs 10
```

### After (HF Streaming)

```bash
python vispr/tools/scripts/train_torch.py \
    --data-source hf_tar_stream \
    --hf-repo username/vispr-dataset \
    --hf-file-path train.tar.gz \
    --hf-val-file-path val.tar.gz \
    --arch resnet50 \
    --epochs 10
```

**No other changes needed!** The rest of the training pipeline remains identical.

## Hybrid Approach

You can mix local and streaming:

```bash
# Stream large training set, use local validation
python vispr/tools/scripts/train_torch.py \
    --data-source hf_tar_stream \
    --hf-repo username/dataset \
    --hf-file-path huge_train.tar.gz \
    --valfile val2017.txt \
    --arch resnet50 \
    --epochs 10
```

Note: Validation will only work with local files in this mode. For full streaming, use `--hf-val-file-path`.

## Best Practices

1. **Archive Organization**
   - Keep archives under 5GB for faster streaming
   - Split large datasets into multiple archives if needed
   - Organize with clear directory structure

2. **Memory Management**
   - Monitor RAM usage during first epoch
   - Adjust buffer size based on available memory
   - Use Colab's high-RAM runtime if needed

3. **Network Reliability**
   - Use stable internet connection
   - Consider downloading for unreliable networks
   - HF Hub has good retry logic built-in

4. **Validation**
   - Use separate validation archive
   - Keep validation set smaller (faster evaluation)
   - No shuffling needed for validation

## Advanced: Custom Dataset Integration

To use streaming with your own custom dataset class:

```python
from data.tar_streaming import HFTarStreamer
from torch.utils.data import IterableDataset

class MyCustomStreamingDataset(IterableDataset):
    def __init__(self, repo_id, file_path):
        self.streamer = HFTarStreamer(repo_id, file_path)

    def __iter__(self):
        for record in self.streamer.extract_structured_data():
            # Custom processing
            image = record['image']
            # ... your transforms ...
            yield processed_image, label
```

## FAQ

**Q: Does streaming slow down training?**
A: First epoch may be slightly slower. Buffer helps maintain throughput. Network speed is the main factor.

**Q: Can I use multiple workers with streaming?**
A: No, `IterableDataset` works best with `num_workers=0`. The module handles this automatically.

**Q: Does it cache downloaded data?**
A: No, to save disk space. Data is re-streamed each epoch. This is intentional for disk-constrained environments.

**Q: Can I use private HF repositories?**
A: Yes, authenticate with `huggingface-cli login` first.

**Q: What happens if network disconnects?**
A: Training will fail. For unreliable networks, use local data loading instead.

**Q: Is the local data path still supported?**
A: Yes! Local loading is the default. Streaming is completely optional.

## Support

- See `data/tar_streaming/README.md` for detailed module documentation
- Check `data/tar_streaming/example_usage.py` for code examples
- Report issues on the GitHub repository

## Summary

**Streaming is perfect when:**
- Using Google Colab or limited disk space
- Dataset is too large for local storage
- Quick experimentation without downloads

**Use local loading when:**
- Unlimited local disk available
- Maximum performance needed
- Unreliable internet connection

Both methods use identical training code - just toggle the data source!
