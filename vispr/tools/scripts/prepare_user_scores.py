"""Prepare user_scores TSV for privacy-aware model training.

Derives per-image per-user privacy scores replicating the original Caffe
PAPInputLayer.forward() (see layers/PAPInputLayer.py:305-321)::

    pool == 'sum':  user_scores_mat = np.dot(img_attr_mat, user_pref_mat)
    pool == 'avg':  same, then divide per-row by attribute cardinality
    pool == 'max':  per-user max of (label * pref) across attributes

The resulting TSV is consumed by the privacy-aware model via --user-scores-path
(see vispr/datasets/pap_dataset.py -- _load_user_scores / _lookup_user_scores).

Output format:
    image_id<TAB>score_0<TAB>score_1<TAB>...<TAB>score_{n_users-1}

- ``image_id`` = os.path.basename(image_path) from the annotation JSON,
  matching the basename fast-path in _lookup_user_scores.
- Header is auto-detected by _load_user_scores when the last cell of the
  first row is non-numeric (so any header row works).

Example usage::

    python -m vispr.tools.scripts.prepare_user_scores \
        --anno-list vispr/datasets/train2017.txt \
        --user-prefs user_studies/user_profiles.tsv \
        --pool max \
        --outfile vispr/datasets/user_scores_train2017.tsv

    python -m vispr.tools.scripts.prepare_user_scores \
        --anno-list vispr/datasets/val2017.txt \
        --user-prefs user_studies/user_profiles.tsv \
        --pool max \
        --outfile vispr/datasets/user_scores_val2017.tsv
"""
import argparse
import csv
import json
import logging
import os
import os.path as osp
import re
import sys

import numpy as np

from vispr import DS_ROOT
from vispr.tools.common.utils import load_attributes, labels_to_vec
from vispr.tools.common.logger import get_logger

logger = get_logger('prepare_user_scores')
# Also route print() through the logger
def _logger_print(*args, **kwargs):
    sep = kwargs.get('sep', ' ')
    end = kwargs.get('end', '\n')
    message = sep.join(map(str, args))
    logger.info(message)
print = _logger_print

_REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
_DEFAULT_OUT_DIR = osp.join(_REPO_ROOT, 'vispr', 'datasets')

_ATTR_ID_RE = re.compile(r'^a\d+_\S+$')

# -------------------------------------------------------------------
# Loaders
# -------------------------------------------------------------------

def _resolve_json_path(entry, ds_root):
    """Return an absolute filesystem path for a single annotation list entry."""
    if osp.isabs(entry):
        return entry
    return osp.join(ds_root, entry)


def load_user_prefs(user_prefs_path, attr_id_to_idx):
    """Parse a user preferences TSV into the preference matrix.

    File format (header auto-detected)::

        <attr_id>  <attr_name>  <score_0>  <score_1>  ...  <score_{U-1}>

    A row is treated as a **data** row when:
      - its first cell matches ``^a\\d+_\\S+$``  (attribute id), and
      - it has >= 3 tab-separated cells, and
      - all cells from index 2 onward are valid floats.

    Rows whose attribute id is not in ``attr_id_to_idx`` are silently
    skipped with a log warning.

    Returns:
        user_pref_mat : np.ndarray  shape (n_attr, U)  float32
        n_users       : int
    """
    n_attr = len(attr_id_to_idx)
    # First pass: determine n_users and which attrs are present
    with open(user_prefs_path, 'r') as f:
        lines = [line.rstrip('\n') for line in f if line.strip()]

    parsed_rows = []          # (attr_id, score_array)
    n_users = None
    skipped_attr_ids = []
    header_line = None

    for line in lines:
        tokens = line.split('\t')
        attr_id = tokens[0]
        # Check if this is a data row
        if len(tokens) < 3 or not _ATTR_ID_RE.match(attr_id):
            header_line = line
            continue
        try:
            scores = np.array([float(x) for x in tokens[2:]], dtype=np.float32)
        except ValueError:
            header_line = line
            continue
        if n_users is None:
            n_users = len(scores)
        if len(scores) != n_users:
            raise ValueError(
                f"Inconsistent number of score columns in user prefs TSV "
                f"(expected {n_users}, got {len(scores)} for {attr_id})."
            )
        if attr_id not in attr_id_to_idx:
            skipped_attr_ids.append(attr_id)
            continue
        parsed_rows.append((attr_id, scores))

    if n_users is None or len(parsed_rows) == 0:
        raise RuntimeError(
            f"No valid data rows found in {user_prefs_path}. "
            f"Expected rows starting with attribute id (e.g. a1_age_approx)."
        )
    if skipped_attr_ids:
        logger.warning(
            "Skipped %d user-prefs attributes not in the model's attribute "
            "list (e.g. %s). These attrs will contribute 0 to privacy scores.",
            len(skipped_attr_ids),
            skipped_attr_ids[:5],
        )
    if header_line is not None:
        logger.info("Detected header line: %s", header_line[:80])

    # Build (n_attr x n_users) matrix
    user_pref_mat = np.zeros((n_attr, n_users), dtype=np.float32)
    for attr_id, scores in parsed_rows:
        user_pref_mat[attr_id_to_idx[attr_id]] = scores

    # Set safe attribute weight to 0 (matches PAPInputLayer SAFE_WEIGHT=0.0)
    if 'a0_safe' in attr_id_to_idx:
        user_pref_mat[attr_id_to_idx['a0_safe']] = 0.0

    logger.info(
        "Loaded user prefs: %d users, %d attributes "
        "(%d with non-zero preference, safe=0).",
        n_users,
        n_attr,
        int(np.count_nonzero(np.any(user_pref_mat != 0, axis=1))),
    )
    return user_pref_mat, n_users


def load_annotations(anno_list_path, attr_id_to_idx, ds_root):
    """Read an annotation list file and return per-image label matrices.

    Each line in the list points to an annotation JSON (relative to ds_root
    or absolute)::

        {"image_path": "...", "labels": ["a1_age_approx", ...]}

    Returns:
        keys       : list[str]        — image_id keys (basename of image_path)
        label_rows : list[np.ndarray] — (68,) float32 binary label vectors
        stats      : dict             — counts of missing labels, skipped rows etc.
    """
    keys = []
    label_rows = []
    stats = {
        'total': 0,
        'loaded': 0,
        'skipped_no_image_path': 0,
        'skipped_no_labels': 0,
        'unknown_attr_ids': {},
    }

    with open(anno_list_path, 'r') as f:
        entries = [line.strip() for line in f if line.strip()]

    stats['total'] = len(entries)
    logger.info("Annotation list: %d entries from %s", len(entries), anno_list_path)

    for entry in entries:
        json_path = _resolve_json_path(entry, ds_root)
        if not osp.exists(json_path):
            logger.warning("Annotation JSON not found (skipped): %s", json_path)
            stats['skipped_no_image_path'] += 1
            continue
        try:
            with open(json_path, 'r') as f:
                anno = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Cannot parse JSON %s: %s", json_path, e)
            stats['skipped_no_image_path'] += 1
            continue

        image_path = anno.get('image_path')
        if not image_path:
            logger.warning("No image_path in %s (skipped).", json_path)
            stats['skipped_no_image_path'] += 1
            continue

        # Collect attribute labels (support both 'labels' list and 'attributes' dict)
        attr_labels = set()
        if 'labels' in anno:
            attr_labels = set(anno['labels'])
        elif 'attributes' in anno:
            for categ_id, attr_list in anno['attributes'].items():
                attr_labels.update(attr_list)

        if not attr_labels:
            stats['skipped_no_labels'] += 1
            # Empty label set → image is 'safe' only
            attr_labels = {'a0_safe'}

        # Track unknown attributes
        known = [a for a in attr_labels if a in attr_id_to_idx]
        unknown = [a for a in attr_labels if a not in attr_id_to_idx]
        for u in unknown:
            stats['unknown_attr_ids'][u] = stats['unknown_attr_ids'].get(u, 0) + 1

        label_vec = np.zeros(len(attr_id_to_idx), dtype=np.float32)
        for attr_id in known:
            label_vec[attr_id_to_idx[attr_id]] = 1.0

        # Ensure at least one active label (safe for empty set)
        if label_vec.sum() == 0:
            if 'a0_safe' in attr_id_to_idx:
                label_vec[attr_id_to_idx['a0_safe']] = 1.0

        image_id = osp.basename(image_path)
        keys.append(image_id)
        label_rows.append(label_vec)
        stats['loaded'] += 1

    if stats['unknown_attr_ids']:
        top5 = sorted(stats['unknown_attr_ids'].items(),
                      key=lambda kv: kv[1], reverse=True)[:5]
        logger.warning(
            "Found %d distinct attribute ids not in the model's attribute "
            "list (ignored in score computation). Top unknowns: %s",
            len(stats['unknown_attr_ids']),
            top5,
        )
    return keys, label_rows, stats


# -------------------------------------------------------------------
# Score computation
# -------------------------------------------------------------------

def build_user_scores_mat(label_mat, user_pref_mat, pool):
    """Replicate PAPInputLayer.user_scores_mat computation.

    Args:
        label_mat    : np.ndarray  shape (N, A)  float32, binary
        user_pref_mat: np.ndarray  shape (A, U)  float32
        pool         : 'sum' | 'avg' | 'max'

    Returns:
        scores       : np.ndarray  shape (N, U)  float32
    """
    label_mat = np.asarray(label_mat, dtype=np.float32)
    N = label_mat.shape[0]
    U = user_pref_mat.shape[1]

    if pool in ('sum', 'avg'):
        scores = np.dot(label_mat, user_pref_mat)           # (N, U)
        if pool == 'avg':
            card = label_mat.sum(axis=1)                    # (N,)
            card = np.where(card > 0, card, 1.0)            # avoid /0
            scores = scores / card[:, None]
        return scores.astype(np.float32)

    # pool == 'max' — chunked to bound memory
    scores = np.zeros((N, U), dtype=np.float32)
    max_per_row_bytes = 8 * 68 * U  # rough upper-bound per row
    chunk_rows = max(1, min(N, int(2**24 // max(1, 68 * U))))

    for start in range(0, N, chunk_rows):
        end = min(N, start + chunk_rows)
        sub = label_mat[start:end, :, None] * user_pref_mat[None, :, :]  # (c, A, U)
        scores[start:end] = sub.max(axis=1)
    return scores


# -------------------------------------------------------------------
# Output writing / validation
# -------------------------------------------------------------------

def write_scores_tsv(keys, scores, outfile):
    """Write the user_scores TSV.

    Header: ``image_id\\tscore_0\\t...\\tscore_{U-1}``
    Data:   ``<image_id>\\t<v0>\\t...\\t<v{U-1}>``
    """
    U = scores.shape[1]
    header = ['image_id'] + [f'score_{i}' for i in range(U)]

    os.makedirs(osp.dirname(outfile) or '.', exist_ok=True)
    with open(outfile, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(header)
        for key, row in zip(keys, scores):
            writer.writerow([key] + [f'{v:.6f}' for v in row])

    logger.info("Wrote %d rows (+ header) to %s", len(keys), outfile)


def validate_output(outfile, expected_rows, n_users, logger=None):
    """Re-read the generated TSV and verify structure.

    Mirrors the header detection logic in PAPDataset._load_user_scores.
    """
    if logger is None:
        logger = logging.getLogger('prepare_user_scores')

    with open(outfile, 'r') as f:
        lines = [line.rstrip('\n') for line in f if line.strip()]

    if len(lines) < 2:
        raise RuntimeError(f"TSV has < 2 lines ({len(lines)}).")

    # Header detection: first row is header iff its last cell fails to parse
    # as float, or it has < 2 cells (same logic as _load_user_scores).
    first_cells = lines[0].split('\t')
    try:
        float(first_cells[-1])
        header_skipped = False
    except (ValueError, IndexError):
        header_skipped = True
        lines = lines[1:]

    if not header_skipped:
        logger.warning("No header row detected (first row's last cell is numeric).")

    if len(lines) != expected_rows:
        raise RuntimeError(
            f"Row count mismatch: expected {expected_rows}, got {len(lines)}."
        )

    errors = []
    for i, line in enumerate(lines, start=(2 if header_skipped else 1)):
        cells = line.split('\t')
        if len(cells) != n_users + 1:
            errors.append(f"Row {i}: expected {n_users+1} cells, got {len(cells)}.")
            continue
        try:
            [float(c) for c in cells[1:]]
        except ValueError:
            errors.append(f"Row {i}: non-numeric score cell(s).")

    if errors:
        for e in errors[:10]:
            logger.error("Validation: %s", e)
        preview = '; '.join(errors[:3])
        raise RuntimeError(
            f"Validation failed with {len(errors)} error(s): {preview}"
        )

    logger.info(
        "Validation passed: %d rows, %d score columns, no format errors.",
        len(lines),
        n_users,
    )


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Prepare user_scores TSV for privacy-aware model training. '
                    'Replicates the score derivation from PAPInputLayer.forward() '
                    '(layers/PAPInputLayer.py:305-321).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--anno-list', required=True,
        help='Annotation list file (one JSON path per line, relative to DS_ROOT).',
    )
    parser.add_argument(
        '--user-prefs', required=True,
        help='User preference TSV (attr_id, attr_name, per-user scores). '
             'Example: user_studies/user_profiles.tsv',
    )
    parser.add_argument(
        '--outfile', default=None,
        help='Output TSV path. Default: vispr/datasets/user_scores_<split>.tsv '
             '(derived from --anno-list basename).',
    )
    parser.add_argument(
        '--pool', choices=['sum', 'avg', 'max'], default='max',
        help='Scoring strategy: '
             "'max' = per-user max over active attributes "
             "(matches original prototxt run, PAPInputLayer default); "
             "'sum' = np.dot(attr_vec, pref_mat) (PAPInputLayer line 311); "
             "'avg' = sum / number_of_active_attributes. Default: max.",
    )
    parser.add_argument(
        '--attr-list', default=None,
        help='Path to attributes.tsv. Default: DS_ROOT/attributes.tsv '
             '(via vispr.tools.common.utils.load_attributes).',
    )
    parser.add_argument(
        '--ds-root', default=None,
        help='Override VISPR_DS_ROOT for resolving relative annotation paths.',
    )
    args = parser.parse_args()

    ds_root = args.ds_root or DS_ROOT

    print(f'prepare_user_scores: loading attributes from {args.attr_list or "DS_ROOT/attributes.tsv"}')
    attr_id_to_name, attr_id_to_idx = load_attributes(args.attr_list)
    n_attr = len(attr_id_to_idx)
    print(f'  {n_attr} model attributes loaded')

    print(f'Loading user preferences from {args.user_prefs}')
    user_pref_mat, n_users = load_user_prefs(args.user_prefs, attr_id_to_idx)
    print(f'  {n_users} user preference columns detected')

    print(f'Loading annotations from {args.anno_list}')
    keys, label_rows, load_stats = load_annotations(args.anno_list, attr_id_to_idx, ds_root)
    print(f'  Annotations loaded: {load_stats["loaded"]} / {load_stats["total"]}')

    if load_stats['skipped_no_image_path']:
        print(f'  Skipped (no image_path or missing JSON): {load_stats["skipped_no_image_path"]}')
    if load_stats['skipped_no_labels']:
        print(f'  Skipped (no labels, treated as safe): {load_stats["skipped_no_labels"]}')
    if load_stats['unknown_attr_ids']:
        print(f'  Unknown attribute ids (ignored): {len(load_stats["unknown_attr_ids"])} distinct')

    if len(keys) == 0:
        logger.error("No annotation rows produced valid output. Aborting.")
        sys.exit(1)

    print(f'Computing privacy scores (pool={args.pool}) ...')
    label_mat = np.stack(label_rows, axis=0)               # (N, n_attr)
    scores = build_user_scores_mat(label_mat, user_pref_mat, args.pool)
    assert scores.shape == (len(keys), n_users)

    # Stats
    score_range = [float(scores.min()), float(scores.max())]
    non_zero_rows = int(np.any(scores != 0, axis=1).sum())
    print(f'  Scores shape: {scores.shape}')
    print(f'  Score range : [{score_range[0]:.4f}, {score_range[1]:.4f}]')
    print(f'  Non-zero rows: {non_zero_rows} / {len(keys)}')

    # Determine output path
    if args.outfile:
        outfile = args.outfile
    else:
        anno_basename = osp.splitext(osp.basename(args.anno_list))[0]
        outfile = osp.join(_DEFAULT_OUT_DIR, f'user_scores_{anno_basename}.tsv')

    print(f'Writing output TSV: {outfile}')
    write_scores_tsv(keys, scores, outfile)

    print(f'Validating output TSV ...')
    validate_output(outfile, len(keys), n_users, logger)
    print('Validation passed.')
    print('Done.')


if __name__ == '__main__':
    main()
