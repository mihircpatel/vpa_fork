# user_scores input data (privacy-aware model)

These TSV files provide per-image, per-user **privacy scores** and are consumed
by the privacy-aware model extension via the training script's
`--user-scores-path` argument (see `vispr/models/privacy_aware_model.py` and
`models/googlenet-prcnn/train_val.prototxt` lines 2121-2220).

## Files in this directory

| File | Split | Rows | Source annotation list |
|------|-------|------|------------------------|
| `user_scores_train2017.tsv` | train | 10,000 | `train2017.txt` |
| `user_scores_val2017.tsv`   | val   | 4,167  | `val2017.txt` |

Each file has a header row:

```
image_id    score_0    score_1    ...    score_29
```

- `image_id` — `os.path.basename(image_path)` of the annotation JSON
  (e.g. `2017_10001018.jpg`). `PAPDataset._lookup_user_scores` matches this as
  its **fast path**.
- `score_0`..`score_29` — the privacy score of the image **for each of the 30
  users**. The number of columns equals the number of users in the preference
  TSV (`user_studies/user_profiles.tsv` → 30 users). Must match
  `--num-privacy-scores` (default 30) when training.

The header is optional / auto-detected: `PAPDataset._load_user_scores`
treats the first row as a header when its last cell is non-numeric.

## How the scores are derived

Replicates the original Caffe data layer
`layers/PAPInputLayer.py` — `PAPMultilabelDataLayerSync.forward()` (lines
305-321). For each image:

1. Build a binary attribute vector `img_attr_vec` (length 68) from the
   annotation's `labels` (attribute ids aligned to `attributes.tsv`).
2. Load the user preference matrix `user_pref_mat` (68 × 30) from the user
   study TSV (`attribute_id`, `attribute_name`, then one score column per user).
3. Combine per image and user using the selected pooling strategy:

```
sum   : score[i,u] = dot(img_attr_vec[i], user_pref_mat[:,u])     # PAPInputLayer line 311
avg   : score[i,u] = sum / |active attributes of image i|
max   : score[i,u] = max over a of (img_attr_vec[i,a] * user_pref_mat[a,u])
```

The `user_scores_train2017.tsv` / `user_scores_val2017.tsv` files were produced
with `--pool max`, matching the original prototxt run (the prototxt does not set
`pool`, and `PAPInputLayer` defaults to `max`).

## Input file formats

**Annotation list** (e.g. `train2017.txt`) — one JSON path per line, relative
to the dataset root:

```
annotations/train2017/2017_10001018.json
```

**Annotation JSON**:

```json
{"image_path": "images/train2017/2017_10001018.jpg",
 "labels": ["a10_face_partial", "a16_race", "..."]}
```

`labels` may reference attributes not in the 68-attribute ground truth; those
are ignored (counted and logged). Images with an empty label set are treated as
`safe` (`a0_safe` only).

**User preferences TSV** (e.g. `user_studies/user_profiles.tsv`) — header
optional; rows are `<attr_id>\t<attr_name>\t<scores...>`:

```
a1_age_approx    Age Group    1.3333    2.2273    ...
```

- `a0_safe` always gets weight 0 (matches `SAFE_WEIGHT = 0.0`).
- Attributes missing from the preference file contribute 0 to all scores.
- Rows whose `attr_id` is not in the 68-attribute list are skipped (logged).

## Regenerating the files

```powershell
python -m vispr.tools.scripts.prepare_user_scores `
  --anno-list vispr/datasets/train2017.txt `
  --user-prefs user_studies/user_profiles.tsv `
  --pool max `
  --outfile vispr/datasets/user_scores_train2017.tsv

python -m vispr.tools.scripts.prepare_user_scores `
  --anno-list vispr/datasets/val2017.txt `
  --user-prefs user_studies/user_profiles.tsv `
  --pool max `
  --outfile vispr/datasets/user_scores_val2017.tsv
```

Options: `--pool {sum,avg,max}`, `--attr-list <attributes.tsv>`,
`--ds-root <override>` (default: `VISPR_DS_ROOT`). The script validates the
output by re-reading it with the same header-detection rules used by
`PAPDataset` and logs progress at each step (logger: `prepare_user_scores`,
written to `logs/prepare_user_scores.log`).

## Usage with the privacy-aware model

```powershell
python vispr/tools/scripts/train_torch.py `
  --infile train2017.txt --valfile val2017.txt `
  --model-type privacy_aware `
  --user-scores-path vispr/datasets/user_scores_train2017.tsv `
  --epochs 10 --save-path ./model_prcnn.pth
```

Note: for streaming data sources only a single `--user-scores-path` is passed
(the same file is used for both train and validation steps if present).