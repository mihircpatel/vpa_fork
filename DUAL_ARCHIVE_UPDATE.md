# Dual Archive Mode Update

## Overview

The HF Tar Streaming module now supports **dual archive mode**, where images and annotations are stored in separate `.tar.gz` files, matching the common dataset organization pattern used in the local `PAPDataset`.

## What's New

### 🎯 Two Streaming Modes

1. **Combined Archive Mode** (Original)
   - Images and annotations in a single `.tar.gz` file
   - Use `--hf-file-path`

2. **Dual Archive Mode** (NEW)
   - Images in one `.tar.gz` file
   - Annotations in another `.tar.gz` file
   - Annotation list in `.txt` file (like local mode)
   - Use `--hf-image-archive`, `--hf-anno-archive`, `--hf-anno-list`

### 📁 Expected Archive Structure

**Dual Archive Mode:**
```
# Image archive (train2017_images.tar.gz)
train2017/
├── 000000000001.jpg
├── 000000000002.jpg
└── ...

# Annotation archive (train2017_annotations.tar.gz)
train2017/
├── 000000000001.json
├── 000000000002.json
└── ...

# Annotation list (train2017.txt)
annotations/train2017/000000000001.json
annotations/train2017/000000000002.json
...
```

## Usage

### Command-Line (Dual Archive Mode)

```bash
python vispr/tools/scripts/train_torch.py \
    --data-source hf_tar_stream \
    --hf-repo username/vispr-dataset \
    --hf-image-archive train2017_images.tar.gz \
    --hf-anno-archive train2017_annotations.tar.gz \
    --hf-anno-list train2017.txt \
    --hf-val-image-archive val2017_images.tar.gz \
    --hf-val-anno-archive val2017_annotations.tar.gz \
    --hf-val-anno-list val2017.txt \
    --arch resnet50 \
    --pretrained \
    --epochs 10
```

### Command-Line (Combined Archive Mode - Original)

```bash
python vispr/tools/scripts/train_torch.py \
    --data-source hf_tar_stream \
    --hf-repo username/vispr-dataset \
    --hf-file-path train.tar.gz \
    --hf-val-file-path val.tar.gz \
    --arch resnet50 \
    --epochs 10
```

### YAML Configuration (Dual Archive Mode)

```yaml
data_source: "hf_tar_stream"

streaming:
  repo_id: "username/vispr-dataset"
  image_archive_path: "train2017_images.tar.gz"
  annotation_archive_path: "train2017_annotations.tar.gz"
  anno_list_path: "train2017.txt"
  buffer_size: 1000

data:
  batch_size: 32
  num_classes: 68
```

### Python API (Dual Archive Mode)

```python
from data.tar_streaming import StreamingConfig, StreamingPAPDataset, HFDualArchiveStreamer

# Option 1: Using StreamingConfig
config = StreamingConfig(
    data_source='hf_tar_stream',
    repo_id='username/vispr-dataset',
    image_archive_path='train2017_images.tar.gz',
    annotation_archive_path='train2017_annotations.tar.gz',
    anno_list_path='train2017.txt',
    buffer_size=1000
)

dataset = StreamingPAPDataset(config=config, shuffle=True)
loader = DataLoader(dataset, batch_size=32)

# Option 2: Using HFDualArchiveStreamer directly
streamer = HFDualArchiveStreamer(
    repo_id='username/vispr-dataset',
    image_archive_path='train2017_images.tar.gz',
    annotation_archive_path='train2017_annotations.tar.gz',
    anno_list_path='train2017.txt'
)

for record in streamer.extract_structured_data():
    image = record['image']  # PIL Image
    labels = record['labels']  # List of labels
    # Process...
```

## New Command-Line Arguments

| Argument | Description |
|----------|-------------|
| `--hf-image-archive` | Path to image .tar.gz in HF repo |
| `--hf-anno-archive` | Path to annotation .tar.gz in HF repo |
| `--hf-anno-list` | Path to annotation list .txt file |
| `--hf-val-image-archive` | Validation image archive |
| `--hf-val-anno-archive` | Validation annotation archive |
| `--hf-val-anno-list` | Validation annotation list |

## Implementation Details

### New Files Created

1. **`data/tar_streaming/dual_archive_streamer.py`**
   - `HFDualArchiveStreamer` class
   - Handles separate image and annotation archives
   - Matches images with annotations by filename
   - Supports annotation list filtering

### Updated Files

2. **`data/tar_streaming/config.py`**
   - Added `image_archive_path`, `annotation_archive_path`, `anno_list_path`
   - Added `is_dual_archive_mode()` method
   - Updated validation logic

3. **`data/tar_streaming/streaming_dataset.py`**
   - Auto-detects dual vs combined archive mode
   - Instantiates appropriate streamer type

4. **`data/tar_streaming/__init__.py`**
   - Exports `HFDualArchiveStreamer`

5. **`vispr/tools/scripts/train_torch.py`**
   - Added dual archive command-line arguments
   - Auto-detects mode based on arguments
   - Supports dual archive validation

6. **`configs/data_config.yaml`**
   - Added dual archive configuration examples
   - Documented both modes

## Backward Compatibility

✅ **100% Backward Compatible**

- Existing combined archive mode works unchanged
- Default behavior unchanged
- All original arguments still supported
- Dual archive mode is opt-in

### Migration Examples

**From Local to Dual Archive Streaming:**

```bash
# Before (Local)
python train_torch.py \
    --infile train2017.txt \
    --valfile val2017.txt \
    --arch resnet50

# After (HF Dual Archive)
python train_torch.py \
    --data-source hf_tar_stream \
    --hf-repo username/dataset \
    --hf-image-archive train2017_images.tar.gz \
    --hf-anno-archive train2017_annotations.tar.gz \
    --hf-anno-list train2017.txt \
    --hf-val-image-archive val2017_images.tar.gz \
    --hf-val-anno-archive val2017_annotations.tar.gz \
    --hf-val-anno-list val2017.txt \
    --arch resnet50
```

## Benefits

✅ **Matches local dataset pattern** - Uses annotation list like `PAPDataset`
✅ **Realistic dataset structure** - Separate archives for images/annotations
✅ **Directory organization** - Handles `train2017/`, `val2017/` structure
✅ **Efficient loading** - Only loads samples from annotation list
✅ **Flexible deployment** - Choose mode based on dataset organization

## Comparison: Combined vs Dual Archive Mode

| Aspect | Combined Archive | Dual Archive |
|--------|------------------|--------------|
| **Files** | 1 `.tar.gz` | 2 `.tar.gz` + 1 `.txt` |
| **Organization** | Mixed | Separated |
| **Annotation List** | No | Yes (like local mode) |
| **Use Case** | Simple datasets | Complex datasets with splits |
| **Match Local Mode** | No | Yes |

## Testing

### Syntax Validation

```bash
python -m py_compile data/tar_streaming/dual_archive_streamer.py
python -m py_compile data/tar_streaming/config.py
python -m py_compile data/tar_streaming/streaming_dataset.py
python -m py_compile vispr/tools/scripts/train_torch.py
```

### Functional Test

```bash
# Test dual archive mode
python vispr/tools/scripts/train_torch.py \
    --data-source hf_tar_stream \
    --hf-repo username/test-dataset \
    --hf-image-archive test_images.tar.gz \
    --hf-anno-archive test_annotations.tar.gz \
    --hf-anno-list test.txt \
    --arch resnet50 \
    --epochs 1 \
    --batch-size 4
```

## Google Colab Example

```python
# Install dependencies
!pip install huggingface_hub PyYAML

# Clone repository
!git clone https://github.com/your-org/vpa_fork.git
%cd vpa_fork

# Train with dual archive streaming
!python vispr/tools/scripts/train_torch.py \
    --data-source hf_tar_stream \
    --hf-repo username/vispr-dataset \
    --hf-image-archive train2017_images.tar.gz \
    --hf-anno-archive train2017_annotations.tar.gz \
    --hf-anno-list train2017.txt \
    --arch resnet50 \
    --pretrained \
    --epochs 5 \
    --batch-size 16
```

## Troubleshooting

### "anno_list_path is required for dual archive mode"

Make sure all three dual archive arguments are provided:
```bash
--hf-image-archive ... \
--hf-anno-archive ... \
--hf-anno-list ...
```

### "Annotation not found in archive"

Check that:
1. Annotation list paths match archive structure
2. Filenames are consistent (basename matching)
3. Archive directory structure is correct

### Mixed mode error

Don't mix combined and dual archive arguments:
```bash
# Wrong ❌
--hf-file-path combined.tar.gz \
--hf-image-archive images.tar.gz

# Right ✅ (pick one mode)
--hf-file-path combined.tar.gz
# OR
--hf-image-archive images.tar.gz --hf-anno-archive annos.tar.gz --hf-anno-list list.txt
```

## Summary

The dual archive mode update makes HF streaming match the local dataset loading pattern, supporting realistic dataset organizations where images and annotations are stored separately. This is especially useful for large-scale datasets organized by splits (train2017, val2017, etc.).

**Key Takeaway:** You can now stream datasets from HuggingFace Hub using the exact same annotation list format as local loading, while maintaining zero disk usage!
