# AGENTS.md

## Project

VPA (Visual Privacy Advisor) - PyTorch re-implementation. Predicts 68 visual attributes per image (multi-label classification, BCEWithLogitsLoss). Originally Caffe-based, migrated to PyTorch.

## Environment

- **Python 3**, Windows (paths use `\` in scripts). Virtual env: `.venv` or `.vpa_venv` (both gitignored).
- `pip install -r requirements.txt` — core deps: `torch`, `torchvision`, `Pillow`, `numpy`, `scikit-learn`, `matplotlib`, `onnxscript`, `huggingface_hub`, `PyYAML`. Optional: `boto3`, `azure-storage-blob`, `gdown`.
- No formal packaging (`pyproject.toml` / `setup.py` absent). No linter, formatter, or typechecker configured. No CI.

## Key Commands

```powershell
# Training (local data)
python vispr\tools\scripts\train_torch.py --infile train.txt --valfile val.txt --epochs 10 --save-path ./model.pth

# Prepare user_scores TSV for privacy-aware model (see vispr/datasets/README_user_scores.md)
python -m vispr.tools.scripts.prepare_user_scores --anno-list vispr\datasets\train2017.txt --user-prefs user_studies\user_profiles.tsv --pool max --outfile vispr\datasets\user_scores_train2017.tsv

# Train (HF streaming with caching)
python vispr\tools\scripts\train_torch.py --data-source hf_tar_stream --hf-repo user/dataset --hf-file-path train.tar.gz --cache-dir ./stream_cache --max-retries 5

# Train (local tar streaming, no extraction needed)
python vispr\tools\scripts\train_torch.py --data-source local_tar_stream --local-file-path ./data/train.tar.gz --epochs 10

# Inference
python vispr\tools\scripts\attribute_predict_torch.py --infile test.txt --weights model.pth --outfile predictions.jsonl

# Evaluate
python vispr\tools\scripts\evaluate.py predictions.jsonl --class_scores metrics.tsv

# ONNX export
python vispr\tools\scripts\export_to_onnx.py --weights model.pth --output model.onnx

# Streaming unit tests (no network required)
python -m pytest tests/test_streaming.py -v -m "not integration"

# Smoke tests (no test framework — just run the script)
python tests\run_unit_tests.py
```

## Architecture

```
vispr/
  __init__.py              # DS_ROOT, CAFFE_ROOT (env-configurable)
  datasets/pap_dataset.py  # PAPDataset (local JSON annotations; optional user_scores_path for privacy-aware training)
  models/                  # Model classes: attribute_model.py (AttributeModel), privacy_aware_model.py (PrivacyAwareAttributeModel)
  torch_utils/transformer.py  # SimpleTransformer (RGB→BGR, mean subtract, HWC→CHW)
  tools/scripts/           # train, inference, evaluate, export scripts
  tools/common/utils.py    # load_attributes(), labels_to_vec()
data/tar_streaming/        # Tar.gz streaming module (HF Hub & local disk, isolated, optional)
configs/data_config.yaml   # data_source toggle: "local" vs "hf_tar_stream" vs "local_tar_stream"
```

## Model classes (--model-type)

- `vispr/models/attribute_model.py` — **`AttributeModel`**: torchvision backbone + 68-dim Linear head. State-dict keys identical to old ResNet checkpoints (`backbone.*`, `backbone.fc.*` — note `backbone` wraps the whole torchvision model, so `fc.weight` appears as `backbone.fc.weight`). Uses `pretrained=` flag which triggers torchvision deprecation warnings (harmless).
- `vispr/models/privacy_aware_model.py` — **`PrivacyAwareAttributeModel(AttributeModel)`**: adds `self.privacy_branch = Sequential(Linear(68,128), Sigmoid, Linear(128,128), Sigmoid, Linear(128, n))` (matches `googlenet-prcnn/train_val.prototxt` lines 2121+). `forward(x)` returns a **tuple** `(attr_logits, privacy_scores)`. Base checkpoints load via `strict=False`.
- Select via `--model-type attribute|privacy_aware` in train/inference/export scripts. Default `attribute` = original behavior.
- Conv-loss note: when loading a base (attribute) checkpoint into a privacy-aware model, `reload_model_weights(..., strict=False)` loads backbone; privacy branch stays random.

## Gotchas

- **`DS_ROOT`** (`vispr/__init__.py:9`): Defaults to `/content/vpa_fork/vispr/datasets/`. Set `VISPR_DS_ROOT` env var to match your data layout, or use absolute paths in annotation files. Evaluation script depends on this for resolving relative annotation paths.
- **`attributes.tsv`** is the ground truth for the 68 attribute classes. `load_attributes()` in `vispr/tools/common/utils.py` reads it relative to `DS_ROOT`.
- **SimpleTransformer** (`vispr/torch_utils/transformer.py`): Preprocessing is Caffe-style — RGB→BGR channel swap, subtract BGR mean `[104, 117, 123]`, HWC→CHW transpose. This is baked into the default transform in `PAPDataset`. Don't use torchvision standard transforms here without understanding the mismatch.
- **Checkpoint format**: Saved as `{'epoch', 'state_dict', 'optimizer'}`. Inference scripts handle both raw state_dict and full checkpoint dict (strips `module.` prefix from DataParallel).
- **Streaming import is soft**: `from data.tar_streaming import ...` is wrapped in try/except. Streaming features are optional; core training/inference works without HuggingFace deps.
- **Logging**: Scripts redirect `print()` to a file logger (`logs/`). Don't rely on stdout for debugging; check `logs/` directory.
- **Data format**: Annotation JSON files need `image_path` (string) and `labels` (list of attribute IDs like `a0_safe`). The list files (`train.txt`, etc.) contain one annotation JSON path per line.

## Annotation JSON format

```json
{"image_path": "relative/path/to/image.jpg", "labels": ["a0_safe", "a3_violence"], "safe": false}
```

Attribute IDs follow the pattern `a{idx}_{name}` where idx is a zero-based integer (68 total, defined in `attributes.tsv`).
