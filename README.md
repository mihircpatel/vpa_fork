# VPA (VisPR Attribute Predictor) - PyTorch Edition

## Overview

This is a complete PyTorch implementation of a multi-label visual attribute prediction system. It predicts 68 visual attributes across images (e.g., safe content, adult content, violence, etc.).

**Based on:** Towards a Visual Privacy Advisor: Understanding and Predicting Privacy Risks in Images  
**Original Project:** https://tribhuvanesh.github.io/vpa/

**Key Features:**
- ✓ PyTorch-based training and inference (no Caffe required)
- ✓ Multi-label classification with BCEWithLogitsLoss  
- ✓ Easy data preparation (JSON annotations)
- ✓ Model validation with mAP metrics
- ✓ Export to ONNX for cross-platform deployment
- ✓ Simple Python API for inference
- ✓ Batch processing capabilities
- ✓ Environment variable configuration (no hardcoded paths)
- ✓ Optional tar.gz streaming from Hugging Face Hub or local disk
- ✓ Adaptive learning rate scheduling (linear warmup + plateau-based reduction)

---

## Quick Start (5 minutes)

### Set PYTHONPATH
```powershell
$env:PYTHONPATH = "$pwd;${env:PYTHONPATH}"
```
### 1. Install

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
pip install --upgrade pip
pip install -r .\\requirements.txt
```

### 2. Prepare Data

Create JSON annotation files with image paths and labels, then create text files listing annotation paths (one per line).

### 3. Train

```powershell
python vispr\tools\scripts\train_torch.py `
    --infile train.txt `
    --valfile val.txt `
    --epochs 10 `
    --save-path ./model.pth
```

#### Learning Rate Scheduling

Training uses **linear warmup** followed by **ReduceLROnPlateau** (drops LR by 1/5 on loss stagnation):

| Argument | Default | Description |
|----------|---------|-------------|
| `--lr` | `1e-3` | Base learning rate |
| `--warmup-epochs` | `5` | Linear warmup from ~0 to `--lr` |
| `--lr-patience` | `3` | Epochs without improvement before LR reduction |
| `--lr-factor` | `0.2` | Factor to multiply LR on plateau (0.2 = drop to 1/5) |
| `--min-lr` | `1e-6` | Minimum learning rate floor |
| `--cooldown` | `0` | Epochs to wait after a reduction before resuming patience |

```powershell
# Example: longer warmup, more aggressive scheduling
python vispr\tools\scripts\train_torch.py `
    --infile train.txt --valfile val.txt `
    --epochs 20 --lr 1e-3 `
    --warmup-epochs 8 --lr-patience 5 --lr-factor 0.2 `
    --save-path ./model.pth
```

**Schedule behavior:**
- **Epochs 1–warmup**: LR ramps linearly from `start_factor × lr` (≈1e-5) to `lr` (1e-3)
- **After warmup**: LR stays constant until loss stagnates for `lr-patience` epochs
- **On plateau**: LR is multiplied by `lr-factor` (default 0.2, i.e., drops to 1/5)
- **Cooldown**: After a reduction, wait `cooldown` epochs before counting patience again
- **Floor**: LR never drops below `min-lr`

### 4. Evaluate

```powershell
python vispr\\tools\\scripts\\attribute_predict_torch.py `
    --infile test.txt `
    --outfile predictions.jsonl `
    --weights ./model.pth

python vispr\\tools\\scripts\\evaluate.py predictions.jsonl
```

### 5. Use Locally

```python
from examples.inference_examples import AttributePredictor
predictor = AttributePredictor('./model.pth')
probs = predictor.predict_image('image.jpg')
print(probs)  # Array of 68 attribute probabilities
```

---

## Complete Documentation

- **TRAINING_AND_DEPLOYMENT_GUIDE.md** — End-to-end step-by-step guide with examples
- **INFERENCE_GUIDE.md** — Inference flow, output format, and data source options
- **examples/inference_examples.py** — Code examples for different use cases
- **requirements.txt** — Python dependencies

---

## Quick Reference

| Task | Command |
|------|---------|
| Train (local files) | `python vispr\tools\scripts\train_torch.py --infile train.txt --valfile val.txt --epochs 20` |
| Train (HF streaming) | `python vispr\tools\scripts\train_torch.py --data-source hf_tar_stream --hf-repo user/dataset --hf-file-path train.tar.gz --epochs 20` |
| Train (local tar streaming) | `python vispr\tools\scripts\train_torch.py --data-source local_tar_stream --local-file-path ./data/train.tar.gz --epochs 20` |
| Train (custom LR schedule) | `python vispr\tools\scripts\train_torch.py --infile train.txt --lr 1e-3 --warmup-epochs 5 --lr-patience 3 --epochs 20` |
| Inference | `python vispr\tools\scripts\attribute_predict_torch.py --infile test.txt --weights model.pth --outfile pred.jsonl` |
| Evaluate | `python vispr\tools\scripts\evaluate.py pred.jsonl --class_scores metrics.tsv` |
| Export ONNX | `python vispr\tools\scripts\export_to_onnx.py --weights model.pth --output model.onnx` |

---

## Next Steps

1. **Read** TRAINING_AND_DEPLOYMENT_GUIDE.md for detailed instructions with code examples
2. **Check** examples/inference_examples.py for usage patterns
3. **Train** your first model on your data
4. **Deploy** the model locally or in production

---

## Migration from Caffe

This repository was recently migrated from Caffe to PyTorch to be more accessible and maintainable.

**What's New:**
- Pure PyTorch training/inference (no Caffe install needed)
- Simpler data API (standard PyTorch Dataset)
- ONNX export support for cross-platform deployment
- Tar.gz streaming from Hugging Face Hub or local disk (no extraction needed)
- Modern Python 3 codebase
- Easy local inference with AttributePredictor wrapper
- Adaptive LR scheduling: linear warmup + ReduceLROnPlateau (drops to 1/5 on stagnation)

**Legacy Caffe features:**
- Original Caffe datalayers preserved but optional (import-safe)
- Weight extraction/conversion helpers provided for existing models

---

**Last Updated:** May 17, 2026
