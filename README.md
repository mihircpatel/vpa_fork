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
- ✓ Privacy-aware model (PRCNN extension) predicting per-image privacy scores in parallel with attribute logits

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
| Train (privacy-aware) | `python vispr\tools\scripts\train_torch.py --infile train.txt --valfile val.txt --model-type privacy_aware --user-scores-path user_scores_train.tsv --val-user-scores-path user_scores_val.tsv --epochs 20` |
| Prepare user_scores | `python -m vispr.tools.scripts.prepare_user_scores --anno-list train2017.txt --user-prefs user_studies\user_profiles.tsv --outfile user_scores.tsv` |
| Inference | `python vispr\tools\scripts\attribute_predict_torch.py --infile test.txt --weights model.pth --outfile pred.jsonl` |
| Inference (privacy-aware) | `python vispr\tools\scripts\attribute_predict_torch.py --infile test.txt --weights model.pth --outfile pred.jsonl --model-type privacy_aware --predict-privacy-scores` |
| Evaluate | `python vispr\tools\scripts\evaluate.py pred.jsonl --class_scores metrics.tsv` |
| Export ONNX | `python vispr\tools\scripts\export_to_onnx.py --weights model.pth --output model.onnx` |
| Export ONNX (privacy-aware) | `python vispr\tools\scripts\export_to_onnx.py --weights model.pth --output model.onnx --model-type privacy_aware --export-privacy-scores` |

---

## Privacy-Aware Model (PRCNN Extension)

The repository ships two model classes in `vispr/models/`:

- **`AttributeModel`** — the base model (torchvision backbone + 68-dim attribute
  classifier). This matches the original behavior exactly; state-dict keys are
  identical to the previous ResNet checkpoints.
- **`PrivacyAwareAttributeModel`** — extends `AttributeModel` with a privacy
  scoring branch that mirrors the layers defined in
  `models/googlenet-prcnn/train_val.prototxt` (lines 2121+):

```
attribute logits (68-dim, from backbone classifier)
   │
   ├─ fc_ps_1: Linear(68 → 128) + Sigmoid
   ├─ fc_ps_2: Linear(128 → 128) + Sigmoid
   └─ fc9:     Linear(128 → 30)      ← privacy scores
```

The privacy branch takes the 68 attribute logits and predicts a 30-dimensional
per-image privacy score vector. During training, the total loss is:

```
loss = BCEWithLogitsLoss(attr_logits, labels)
     + privacy_loss_weight * MSELoss(privacy_scores, user_scores)
```

with `privacy_loss_weight = 0.03` (matching the prototxt `loss_weight: 0.03`).

### Selecting the model

Use `--model-type privacy_aware` in `train_torch.py`, `attribute_predict_torch.py`,
and `export_to_onnx.py`. The default (`attribute`) keeps the original behavior.

```powershell
# Train: attribute loss + privacy score loss (MSE, weight 0.03)
# Independent user-scores TSVs for the train and validation flows:
python vispr\tools\scripts\train_torch.py `
    --infile train.txt --valfile val.txt `
    --model-type privacy_aware `
    --user-scores-path user_scores_train.tsv `
    --val-user-scores-path user_scores_val.tsv `
    --epochs 20 --save-path ./model_prcnn.pth
```

- `--user-scores-path` targets the **training** split.
- `--val-user-scores-path` targets the **validation** split; it is independent
  and optional (the validation step reports only attribute metrics, so if it is
  omitted the validation dataset is created without user scores).

### User scores TSV format

`--user-scores-path` points to a TSV mapping each image to its 30 privacy score
targets. Format (header optional):

```
image_id  score_0  score_1  ...  score_29
img123.jpg  0.05  0.1  0.2  ...  0.3
img456.jpg  0.1   0.2  0.15 ...  0.25
```

The first column may be the image filename (matched against the annotation's
`image_path` basename), a full image path, or the annotation path. Samples
without a match fall back to a zero vector. If no `--user-scores-path` is given,
the privacy loss is skipped and only the attribute (BCE) loss drives training.
Training and validation use **independent** files selected by
`--user-scores-path` (train split) and `--val-user-scores-path` (validation
split, optional).

### Preparing the user_scores input file

The `user_scores` targets are generated from the annotation labels and the user
study preference scores, replicating the original Caffe `PAPInputLayer.forward()`
(`layers/PAPInputLayer.py:305-321`):

```powershell
# One file per split — each is passed independently below
python -m vispr.tools.scripts.prepare_user_scores `
    --anno-list vispr\datasets\train2017.txt `
    --user-prefs user_studies\user_profiles.tsv `
    --pool max `
    --outfile vispr\datasets\user_scores_train2017.tsv

python -m vispr.tools.scripts.prepare_user_scores `
    --anno-list vispr\datasets\val2017.txt `
    --user-prefs user_studies\user_profiles.tsv `
    --pool max `
    --outfile vispr\datasets\user_scores_val2017.tsv
```

- `--pool`: `sum` (dot product), `avg` (normalized by attribute count), or
  `max` (per-user max; matches the original prototxt run). Default `max`.
- Output TSV has header `image_id score_0 ... score_{U-1}` where `U` = number of
  users in the preference file (30 for `user_studies/user_profiles.tsv`).
- The script loads and validates every annotation, logs each step to
  `logs/prepare_user_scores.log`, and re-validates the written TSV using the
  same header-detection rules as `PAPDataset`.
- See `vispr/datasets/README_user_scores.md` for the full format documentation.
  Shipped files: `vispr/datasets/user_scores_train2017.tsv` (10,000 rows) and
  `vispr/datasets/user_scores_val2017.tsv` (4,167 rows).

### Inference output

With `--model-type privacy_aware --predict-privacy-scores`, the output JSONL
includes both attribute probabilities and privacy scores:

```json
{"anno_path": "...", "pred_probs": [0.1, 0.9, ...], "privacy_scores": [0.05, 0.1, ...]}
```

Without `--predict-privacy-scores`, only `pred_probs` is written, so the output
remains compatible with `evaluate.py`.

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
