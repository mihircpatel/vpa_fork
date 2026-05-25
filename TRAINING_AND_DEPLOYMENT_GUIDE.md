"""Complete guide for setting up, training, evaluating, and deploying the PyTorch attribute prediction model."""

# Complete VPA PyTorch Workflow Guide

## Quick Start

### 1. Environment Setup

```powershell
# Create virtual environment
python -m venv .venv

# Activate
.\\.venv\\Scripts\\Activate.ps1

# Install dependencies
pip install --upgrade pip
pip install -r .\\requirements.txt
```

### 2. Prepare Your Data

Your annotations should be in JSON format. Each annotation file should contain:
```json
{
  "image_path": "path/to/image.jpg",
  "labels": ["a1_attribute", "a2_another_attr"],
  "safe": true
}
```

Create text files listing annotation file paths (one per line):
- `train_list.txt` — paths to training annotations
- `val_list.txt` — paths to validation annotations  
- `test_list.txt` — paths to test annotations

Example:
```
/path/to/anno1.json
/path/to/anno2.json
/path/to/anno3.json
```

### 3. Train the Model

```powershell
python vispr\\tools\\scripts\\train_torch.py `
    --infile train_list.txt `
    --valfile val_list.txt `
    --arch resnet50 `
    --pretrained `
    --epochs 20 `
    --batch-size 32 `
    --lr 0.001 `
    --num-classes 68 `
    --save-path ./checkpoints/model_final.pth
```

**Training Parameters:**
- `--infile`: path to training annotation list
- `--valfile`: path to validation annotation list (optional)
- `--arch`: backbone architecture (default: resnet50)
- `--pretrained`: use ImageNet pretrained weights
- `--epochs`: number of training epochs
- `--batch-size`: batch size for training
- `--lr`: learning rate (Adam optimizer)
- `--num-classes`: number of attributes to predict
- `--save-path`: where to save checkpoints

**Output:**
- `model_last.pth` — checkpoint after final epoch
- `model_best.pth` — checkpoint with best validation mAP (if validation set provided)

### 4. Evaluate the Model

#### A. Run inference on test set

```powershell
python vispr\\tools\\scripts\\attribute_predict_torch.py `
    --infile test_list.txt `
    --outfile predictions.jsonl `
    --weights ./checkpoints/model_best.pth `
    --arch resnet50 `
    --num-classes 68 `
    --batch-size 32
```

**Output:** `predictions.jsonl` containing per-image predictions in format:
```json
{"anno_path": "/path/to/anno1.json", "pred_probs": [0.95, 0.12, ..., 0.43]}
{"anno_path": "/path/to/anno2.json", "pred_probs": [0.05, 0.87, ..., 0.21]}
```

#### B. Compute metrics (mAP, per-class AP)

```powershell
python vispr\\tools\\scripts\\evaluate.py predictions.jsonl `
    --class_scores per_class_metrics.tsv `
    --qual ./qualitative_results
```

**Output:**
- Per-class Average Precision (AP) in `per_class_metrics.tsv`
- Macro mean average precision (mAP) printed to console
- `./qualitative_results/` — (optional) visualizations showing top predictions

Output format:
```
attribute_id    attribute_name    num_occurrences    ap
a0_safe         safe              1000               92.5
a1_adult        adult content     500                87.3
...
```

### 5. Package the Model

#### Option A: Save PyTorch Model (Simple)
The model checkpoint already includes the full state_dict and is ready to use:

```powershell
# Copy model to a package directory
mkdir -p ./model_package
copy ./checkpoints/model_best.pth ./model_package/model.pth

OR

# Copy model (Windows PowerShell)
Copy-Item ./checkpoints/model_final_best.pth ./model_package/model.pth
```

#### Option B: Export to ONNX (For cross-platform deployment)

```powershell
python vispr\tools\scripts\export_to_onnx.py `
    --weights ./checkpoints/model_final_best.pth `
    --arch resnet50 `
    --num-classes 68 `
    --output ./model_package/model.onnx
```

#### Option C: Create a complete package with metadata

Create `./model_package/metadata.json`:
```json
{
  "model_name": "VPA Attribute Predictor",
  "architecture": "resnet50",
  "num_classes": 68,
  "input_shape": [1, 3, 224, 224],
  "input_mean": [104, 117, 123],
  "input_scale": 1.0,
  "channel_order": "BGR",
  "class_names": ["safe", "adult", "..."],
  "date_trained": "2026-05-17",
  "metrics": {
    "train_loss": 0.15,
    "val_map": 0.92
  }
}
```
```powershell
# Create metadata
@"
{
  "model_name": "VPA Attribute Predictor v1.0",
  "architecture": "resnet50",
  "num_classes": 68,
  "input_shape": [1, 3, 224, 224],
  "input_mean": [104.0, 117.0, 123.0],
  "input_scale": 1.0,
  "channel_order": "RGB (converted to BGR internally)",
  "training_date": "2026-05-18",
  "pytorch_version": "1.8.0",
  "metrics": {
    "val_map": 0.8934,
    "test_map": 0.8712
  },
  "attributes": 68,
  "usage": "See examples/inference_examples.py"
}
"@ | Out-File ./model_package/metadata.json -Encoding UTF8
```

### 6. Deploy and Use the Model Locally

#### A. Python API (Inference)

```python
import torch
import torchvision.models as models
from vispr.torch_utils.transformer import SimpleTransformer
from PIL import Image
import numpy as np

# Load model
def load_model(weights_path, num_classes=68):
    model = models.resnet50(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    checkpoint = torch.load(weights_path, map_location='cpu')
    
    # Handle checkpoint vs state_dict
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model

# Preprocess image
def preprocess_image(image_path):
    transformer = SimpleTransformer(mean=[104, 117, 123])
    img = Image.open(image_path).convert('RGB')
    arr = np.asarray(img)
    return transformer.preprocess(arr)

# Run inference
def predict(model, image_path, device='cpu'):
    model = model.to(device)
    with torch.no_grad():
        img_tensor = torch.from_numpy(preprocess_image(image_path)).unsqueeze(0).float()
        img_tensor = img_tensor.to(device)
        logits = model(img_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    return probs

# Usage
model = load_model('./model_package/model.pth')
probs = predict(model, 'test_image.jpg')
print('Attribute probabilities:', probs)
```
```python
#!/usr/bin/env python
"""Example: Run attribute prediction on images."""

import sys
sys.path.insert(0, r'C:\path\to\vpa')  # Update path

from examples.inference_examples import AttributePredictor
from pathlib import Path
import numpy as np
import json

# Initialize predictor (one-time cost)
predictor = AttributePredictor(
    './model_package/model.pth',
    num_classes=68,
    device='cuda'  # Use 'cpu' if no GPU
)

# Single image
print("=" * 50)
print("Single Image Prediction")
print("=" * 50)

probs = predictor.predict_image('image.jpg')
print(f"Shape: {probs.shape}")  # (68,)
print(f"Min prob: {probs.min():.4f}, Max prob: {probs.max():.4f}")
print(f"Top 5 classes: {np.argsort(-probs)[:5]}")

# Batch processing
print("\n" + "=" * 50)
print("Batch Processing")
print("=" * 50)

image_paths = list(Path('./images').glob('*.jpg'))[:100]
predictions = {}

for idx, (img_path, pred_probs) in enumerate(predictor.predict_images(image_paths, batch_size=32)):
    predictions[str(img_path)] = pred_probs.tolist()
    if (idx + 1) % 20 == 0:
        print(f"Processed {idx + 1}/{len(image_paths)} images")

print(f"✓ Processed {len(predictions)} images")

# Save to JSON
with open('batch_predictions.json', 'w') as f:
    json.dump(predictions, f)
print(f"✓ Saved predictions to batch_predictions.json")
```

#### B. Command-line Interface (Single Image)

```powershell
# Create a single-image test list
echo "c:\\path\\to\\anno_temp.json" > test_single.txt

OR

@"
{
  "image_path": "C:\\images\\photo.jpg",
  "labels": [],
  "safe": true
}
"@ | Out-File anno_temp.json

# Create list file
@"
C:\anno_temp.json
"@ | Out-File test_single.txt

AND

# Run inference
python vispr\tools\scripts\attribute_predict_torch.py `
    --infile test_single.txt `
    --outfile pred_single.jsonl `
    --weights ./model_package/model.pth `
    --batch-size 1

# View result
cat pred_single.jsonl | python -m json.tool
```

Output in `result.jsonl`:
```json
{"anno_path": "c:/path/to/anno.json", "pred_probs": [0.95, 0.12, ..., 0.43]}
```

#### C. Batch Processing Script

Create `batch_predict.py`:
```python
import json
import argparse
from pathlib import Path
import torch
import torchvision.models as models
from torch.utils.data import DataLoader
from vispr.datasets.pap_dataset import PAPDataset

def batch_predict(anno_list, weights_path, output_path, batch_size=32, num_classes=68):
    # Load model
    model = models.resnet50(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    dataset = PAPDataset(anno_list, im_shape=(224, 224))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    with open(output_path, 'w') as wf:
        idx = 0
        for images, _ in loader:
            with torch.no_grad():
                images = images.to(device).float()
                logits = model(images)
                probs = torch.sigmoid(logits).cpu().numpy()
            
            for b in range(probs.shape[0]):
                anno_path = dataset.anno_paths[idx]
                entry = {'anno_path': anno_path, 'pred_probs': probs[b].tolist()}
                wf.write(json.dumps(entry) + '\n')
                idx += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('anno_list', help='Path to annotation list')
    parser.add_argument('--weights', required=True)
    parser.add_argument('--output', default='output.jsonl')
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    batch_predict(args.anno_list, args.weights, args.output, args.batch_size)
```

Run:
```powershell
python batch_predict.py test_list.txt --weights ./model_package/model.pth --output predictions.jsonl
```

### 7. Complete End-to-End Example

```powershell
# 1. Setup
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
pip install -r .\\requirements.txt

# 2. Train
python vispr\\tools\\scripts\\train_torch.py `
    --infile train_list.txt `
    --valfile val_list.txt `
    --arch resnet50 `
    --pretrained `
    --epochs 20 `
    --batch-size 32 `
    --num-classes 68 `
    --save-path ./checkpoints/model_final.pth

# 3. Inference
python vispr\\tools\\scripts\\attribute_predict_torch.py `
    --infile test_list.txt `
    --outfile predictions.jsonl `
    --weights ./checkpoints/model_best.pth `
    --arch resnet50 `
    --num-classes 68 `
    --batch-size 32

# 4. Evaluate
python vispr\\tools\\scripts\\evaluate.py predictions.jsonl `
    --class_scores metrics.tsv

# 5. Package
mkdir model_package
copy ./checkpoints/model_best.pth ./model_package/model.pth

# Summary
Write-Host "✓ Training complete!"
Write-Host "✓ Inference complete! Results in: predictions.jsonl"
Write-Host "✓ Metrics: metrics.tsv"
Write-Host "✓ Model packaged in: model_package/"
```

### 8. Deployment Checklist

- [ ] Model checkpoint saved (`model_best.pth`)
- [ ] Test metrics computed (mAP, per-class AP)
- [ ] Model packaged with metadata
- [ ] Local inference tested on sample images
- [ ] Batch processing validated
- [ ] Documentation updated with model details
- [ ] Optional: export to ONNX for cross-platform use
- [ ] Optional: containerize for deployment (Docker)

### 9. Troubleshooting

**Q: Out of memory during training**
A: Reduce `--batch-size` (try 16 or 8)

**Q: No improvement in validation mAP**
A: Train longer with `--epochs 50`, or adjust learning rate with `--lr 0.0001`

**Q: Slow inference**
A: Use GPU device (set `CUDA_VISIBLE_DEVICES=0`), or reduce batch size for higher throughput

**Q: Model predictions are random**
A: Ensure you're loading the correct checkpoint; check that image paths in annotations are correct

### 10. Advanced: Fine-tuning on New Data

```powershell
# Load pretrained model and train on new data
python vispr\\tools\\scripts\\train_torch.py `
    --infile my_new_train.txt `
    --valfile my_new_val.txt `
    --arch resnet50 `
    --weights ./model_package/model.pth `
    --epochs 5 `
    --lr 0.00001 `
    --batch-size 16 `
    --num-classes 68 `
    --save-path ./checkpoints/model_finetuned.pth
```

### Key Differences:
- `--weights` loads pre-trained checkpoint 
- `--lr` is much smaller (0.00001 vs 0.001) to avoid catastrophic forgetting
- `--epochs` is smaller (5 vs 50) since we're transferring knowledge

### Distributed Training (Multi GPU)
```powershell
# Set visible GPUs
$env:CUDA_VISIBLE_DEVICES = "0,1"

# Training uses DataParallel automatically if multiple GPUs detected
python vispr\tools\scripts\train_torch.py `
    --infile train_list.txt `
    --batch-size 128 `
    --epochs 50 `
    --device cuda
```

### Mixed Precision Training (Faster on RTX/A100 GPUs)
To make training ~2x faster on modern GPUs, see `train_torch.py` and look for `torch.cuda.amp` (Automatic Mixed Precision). 

### Data Format Examples
#### Example 1: Simple Binary Labels
```json
{
  "image_path": "/dataset/img_001.jpg",
  "labels": ["a0_safe"],
  "safe": true
}
```

#### Example 2: Multi-label
```json
{
  "image_path": "/dataset/img_002.jpg",
  "labels": ["a2_quality_blur", "a5_weather"],
  "safe": true,
  "metadata": {
    "source": "webcam",
    "confidence": 0.95
  }
}
```

#### Example 3: From Attributes Dict
```json
{
  "image_path": "/dataset/img_003.jpg",
  "attributes": {
    "visual": ["a0_safe", "a2_quality"],
    "context": ["a10_outdoor"]
  },
  "safe": true
}
```
---

### Key File References 
- `vispr/datasets/pap_dataset.py` — Data Loading
- `vispr/torch_utils/transformer.py` — Image Preprocessing
- `vispr/tools/scripts/train_torch.py` — Training
- `vispr/tools/scripts/attribute_predict_torch.py` — Inference
- `vispr/tools/scripts/evaluate.py` - Evaluation and Metrics
- `examples/inference_examples.py` - Python API wrapper
- `vispr/tools/scripts/export_to_onnx.py` - ONNX export (export model for cross environment deployment)