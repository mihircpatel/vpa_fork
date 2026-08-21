# Quick Start: HuggingFace Tar Streaming

## TL;DR - Stream datasets from HuggingFace Hub without downloading!

### 1️⃣ Install (one time)

```bash
pip install huggingface_hub PyYAML
```

### 2️⃣ Train with streaming

```bash
python vispr/tools/scripts/train_torch.py \
    --data-source hf_tar_stream \
    --hf-repo username/your-dataset \
    --hf-file-path train.tar.gz \
    --arch resnet50 \
    --pretrained \
    --epochs 10
```

### 3️⃣ That's it! 🎉

No downloads, no disk space used!

---

## Command Comparison

### Before (Local)
```bash
python vispr/tools/scripts/train_torch.py \
    --infile train2017.txt \
    --arch resnet50 \
    --epochs 10
```

### After (Streaming)
```bash
python vispr/tools/scripts/train_torch.py \
    --data-source hf_tar_stream \
    --hf-repo username/dataset \
    --hf-file-path train.tar.gz \
    --arch resnet50 \
    --epochs 10
```

---

## All Streaming Arguments

| Argument | Required? | Example |
|----------|-----------|---------|
| `--data-source hf_tar_stream` | ✅ Yes | `hf_tar_stream` |
| `--hf-repo` | ✅ Yes | `username/dataset-name` |
| `--hf-file-path` | ✅ Yes | `train.tar.gz` |
| `--hf-val-file-path` | ❌ No | `val.tar.gz` |
| `--buffer-size` | ❌ No | `1000` (default) |

---

## Google Colab

```python
# Install
!pip install huggingface_hub PyYAML

# Clone repo
!git clone https://github.com/your-org/vpa_fork.git
%cd vpa_fork

# Train (no disk space used!)
!python vispr/tools/scripts/train_torch.py \
    --data-source hf_tar_stream \
    --hf-repo username/dataset \
    --hf-file-path train.tar.gz \
    --arch resnet50 \
    --pretrained \
    --epochs 5 \
    --batch-size 16
```

---

## Python API

```python
from data.tar_streaming import StreamingConfig, StreamingPAPDataset
from torch.utils.data import DataLoader

config = StreamingConfig(
    data_source='hf_tar_stream',
    repo_id='username/dataset',
    file_path='train.tar.gz'
)

dataset = StreamingPAPDataset(config=config, shuffle=True)
loader = DataLoader(dataset, batch_size=32)

for images, labels in loader:
    # Your code here
    pass
```

---

## Troubleshooting

**"Module not found: huggingface_hub"**
```bash
pip install huggingface_hub PyYAML
```

**"Out of Memory"**
```bash
--buffer-size 500 --batch-size 16
```

**"repo_id is required"**
```bash
# Make sure to provide both:
--hf-repo username/dataset --hf-file-path train.tar.gz
```

---

## Learn More

- 📖 **Full Guide**: `HF_STREAMING_GUIDE.md`
- 🔧 **Module Docs**: `data/tar_streaming/README.md`
- 💡 **Examples**: `data/tar_streaming/example_usage.py`
- 📋 **Implementation**: `IMPLEMENTATION_SUMMARY.md`

---

## Need Help?

1. Check `HF_STREAMING_GUIDE.md` for detailed instructions
2. See `data/tar_streaming/example_usage.py` for code examples
3. Report issues on GitHub

**Happy streaming! 🚀**
