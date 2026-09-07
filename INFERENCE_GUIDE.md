# Inference Guide

Run attribute predictions using a trained PyTorch model. This script computes per-attribute probabilities for images and outputs results as JSONL.

## Quick Start

```powershell
# Basic inference (local data)
python vispr\tools\scripts\attribute_predict_torch.py `
    --infile test.txt `
    --weights model.pth `
    --outfile predictions.jsonl

# Inference from HF tar streaming
python vispr\tools\scripts\attribute_predict_torch.py `
    --data-source hf_tar_stream `
    --hf-repo username/dataset `
    --hf-file-path test.tar.gz `
    --weights model.pth `
    --outfile predictions.jsonl

# Inference from local tar streaming
python vispr\tools\scripts\attribute_predict_torch.py `
    --data-source local_tar_stream `
    --local-file-path ./data/test.tar.gz `
    --weights model.pth `
    --outfile predictions.jsonl
```

## Input Formats

### Local mode (`--data-source local`)
Provide a text file with one annotation JSON path per line:
```
/path/to/annotations/test2017/2017_10000580.json
/path/to/annotations/test2017/2017_10000581.json
...
```

Each JSON annotation file has the structure:
```json
{
  "image_path": "images/test2017/2017_10000580.jpg",
  "labels": ["a0_safe", "a3_height_approx"],
  "safe": true
}
```

### HF tar streaming mode (`--data-source hf_tar_stream`)
Stream directly from a Hugging Face repository:
```powershell
# Combined archive (images + annotations in one .tar.gz)
python vispr\tools\scripts\attribute_predict_torch.py `
    --data-source hf_tar_stream `
    --hf-repo username/dataset `
    --hf-file-path test.tar.gz `
    --weights model.pth `
    --outfile predictions.jsonl

# Dual archive (separate image and annotation .tar.gz)
python vispr\tools\scripts\attribute_predict_torch.py `
    --data-source hf_tar_stream `
    --hf-repo username/dataset `
    --hf-image-archive test_images.tar.gz `
    --hf-anno-archive test_annotations.tar.gz `
    --hf-anno-list test.txt `
    --weights model.pth `
    --outfile predictions.jsonl
```

### Local tar streaming mode (`--data-source local_tar_stream`)
Stream directly from local `.tar.gz` archives without extracting:
```powershell
# Combined archive
python vispr\tools\scripts\attribute_predict_torch.py `
    --data-source local_tar_stream `
    --local-file-path ./data/test.tar.gz `
    --weights model.pth `
    --outfile predictions.jsonl

# Dual archive
python vispr\tools\scripts\attribute_predict_torch.py `
    --data-source local_tar_stream `
    --local-image-archive ./data/test_images.tar.gz `
    --local-anno-archive ./data/test_annotations.tar.gz `
    --local-anno-list ./data/test_anno_list.txt `
    --weights model.pth `
    --outfile predictions.jsonl
```

## Output Format

The output is a JSONL file (one JSON object per line). Each entry contains:
- `anno_path`: Path to the annotation JSON file
- `pred_probs`: Array of 68 float probabilities (one per attribute)

Example output:
```json
{"anno_path": "C:/data/annotations/test2017/2017_10000580.json", "pred_probs": [0.95, 0.12, 0.03, ...]}
{"anno_path": "C:/data/annotations/test2017/2017_10000581.json", "pred_probs": [0.88, 0.45, 0.01, ...]}
```

The `anno_path` values correspond to the annotation JSON file paths from the input:
- **Local mode**: Full paths as listed in the input text file
- **HF streaming mode**: Archive-relative paths (e.g., `annotations/test2017/2017_10000580.json`)
- **Local tar streaming mode**: Archive-relative paths (same as HF streaming)

## CLI Arguments

### Required
| Argument | Description |
|----------|-------------|
| `--weights` | Path to model weights (.pth) |
| `--outfile` | Output JSONL file path |

### Model
| Argument | Default | Description |
|----------|---------|-------------|
| `--arch` | `resnet50` | Model architecture (resnet18/34/50/101/152, mobilenet_v2) |
| `--num-classes` | 68 | Number of attribute classes |
| `--pretrained` | false | Use pretrained backbone weights |
| `--device` | auto | Device: `cuda` or `cpu` |

### Data source
| Argument | Default | Description |
|----------|---------|-------------|
| `--data-source` | `local` | `local`, `hf_tar_stream`, or `local_tar_stream` |
| `--infile` | None | Annotation list file (required for local mode) |
| `--config` | None | Path to YAML config file |

### HF streaming
| Argument | Description |
|----------|-------------|
| `--hf-repo` | HuggingFace repository ID |
| `--hf-file-path` | Combined .tar.gz path in repo |
| `--hf-image-archive` | Image .tar.gz path in repo (dual mode) |
| `--hf-anno-archive` | Annotation .tar.gz path in repo (dual mode) |
| `--hf-anno-list` | Annotation list .txt path (dual mode) |

### Local tar streaming
| Argument | Description |
|----------|-------------|
| `--local-file-path` | Local combined .tar.gz path |
| `--local-image-archive` | Local image .tar.gz path (dual mode) |
| `--local-anno-archive` | Local annotation .tar.gz path (dual mode) |
| `--local-anno-list` | Local annotation list .txt path (dual mode) |

### Streaming settings
| Argument | Default | Description |
|----------|---------|-------------|
| `--batch-size` | 64 | Batch size for inference |
| `--buffer-size` | 1000 | Shuffle buffer size for streaming |
| `--chunk-size` | 8MB | Read chunk size for tar streaming |
| `--cache-dir` | None | Local cache directory for streaming records |
| `--log-interval` | 100 | Log progress every N records |
| `--max-retries` | 3 | Max retries on network errors (HF mode) |

## Evaluation

After inference, evaluate predictions with:
```powershell
python vispr\tools\scripts\evaluate.py predictions.jsonl --class_scores metrics.tsv
```

## Examples

### Batch inference on large dataset
```powershell
python vispr\tools\scripts\attribute_predict_torch.py `
    --arch resnet50 `
    --weights model.pth `
    --data-source local_tar_stream `
    --local-file-path ./data/test_large.tar.gz `
    --outfile predictions.jsonl `
    --batch-size 128 `
    --device cuda
```

### Inference with caching (for repeated runs)
```powershell
python vispr\tools\scripts\attribute_predict_torch.py `
    --data-source hf_tar_stream `
    --hf-repo username/dataset `
    --hf-file-path test.tar.gz `
    --weights model.pth `
    --outfile predictions.jsonl `
    --cache-dir ./inference_cache
```
