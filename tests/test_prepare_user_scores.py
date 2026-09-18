"""Unit tests for vispr.tools.scripts.prepare_user_scores.

Verifies the user_scores TSV generation logic replicates the Caffe
PAPInputLayer.forward() derivation (layers/PAPInputLayer.py:305-321).
"""
import json
import os
import sys

import numpy as np
import pytest

from vispr.tools.scripts.prepare_user_scores import (
    build_user_scores_mat,
    load_annotations,
    load_user_prefs,
    validate_output,
    write_scores_tsv,
)

# A small (3-attribute, 2-user) model attribute map used for tests
ATTR_ID_TO_IDX = {"a0_safe": 0, "a1_age_approx": 1, "a2_weight_approx": 2}


def _write_anno(tmp_path, image_path, labels):
    anno_path = tmp_path / ('a' + str(abs(hash(image_path))) + '.json')
    anno_path.write_text(json.dumps({"image_path": image_path, "labels": labels}))
    return str(anno_path)


def _write_user_prefs(tmp_path, header=False):
    path = tmp_path / 'user_prefs.tsv'
    lines = []
    if header:
        lines.append('attribute_id\tattribute_name')
    lines.append('a0_safe\tSafe\t0.0\t0.0')
    lines.append('a1_age_approx\tAge Group\t1.0\t2.0')
    lines.append('a2_weight_approx\tWeight Group\t3.0\t4.0')
    lines.append('a999_unknown\tControl\t5.0\t5.0')
    path.write_text('\n'.join(lines) + '\n')
    return str(path)


@pytest.mark.parametrize('header', [False, True])
def test_load_user_prefs_shape_and_headers(tmp_path, header):
    path = _write_user_prefs(tmp_path, header=header)
    mat, n_users = load_user_prefs(path, ATTR_ID_TO_IDX)
    assert mat.shape == (3, 2)
    assert n_users == 2
    np.testing.assert_allclose(mat[0], [0.0, 0.0])  # safe -> 0
    np.testing.assert_allclose(mat[1], [1.0, 2.0])
    np.testing.assert_allclose(mat[2], [3.0, 4.0])


def test_load_user_prefs_inconsistent_width_raises(tmp_path):
    path = tmp_path / 'bad.tsv'
    path.write_text('a1_age_approx\tAge Group\t1.0\t2.0\n'
                    'a2_weight_approx\tWeight Group\t3.0\n')
    with pytest.raises(ValueError, match='Inconsistent'):
        load_user_prefs(str(path), ATTR_ID_TO_IDX)


def test_load_user_prefs_no_rows_raises(tmp_path):
    path = tmp_path / 'empty.tsv'
    path.write_text('attribute_id\tattribute_name\n')
    with pytest.raises(RuntimeError, match='No valid data rows'):
        load_user_prefs(str(path), ATTR_ID_TO_IDX)


@pytest.mark.parametrize('pool', ['sum', 'avg', 'max'])
def test_build_user_scores_mat_matches_reference(pool):
    label_mat = np.array([[1, 0, 1],
                          [0, 1, 0],
                          [1, 1, 0]], dtype=np.float32)
    pref = np.array([[1.0, 2.0],
                     [2.0, 1.0],
                     [0.5, 0.5]], dtype=np.float32)

    if pool == 'sum':
        expected = label_mat @ pref
    elif pool == 'avg':
        expected = (label_mat @ pref) / label_mat.sum(axis=1)[:, None]
    else:
        expected = (label_mat[:, :, None] * pref[None, :, :]).max(axis=1)

    out = build_user_scores_mat(label_mat, pref, pool)
    assert out.shape == (3, 2)
    np.testing.assert_allclose(out, expected, rtol=1e-6)


def test_build_user_scores_mat_avg_zero_cardinality():
    # Empty (all-zero) label rows must not produce NaN/Inf in avg mode
    label_mat = np.zeros((2, 3), dtype=np.float32)
    pref = np.ones((3, 2), dtype=np.float32)
    out = build_user_scores_mat(label_mat, pref, 'avg')
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out, 0.0)


def test_load_annotations_basic(tmp_path):
    anno = _write_anno(tmp_path, 'images/train/2017_10001018.jpg',
                       ['a1_age_approx', 'a2_weight_approx'])
    list_path = tmp_path / 'list.txt'
    list_path.write_text(os.path.basename(anno) + '\n')

    keys, label_rows, stats = load_annotations(str(list_path), ATTR_ID_TO_IDX,
                                               ds_root=str(tmp_path))
    assert stats['total'] == 1 and stats['loaded'] == 1
    assert keys == ['2017_10001018.jpg']
    np.testing.assert_allclose(label_rows[0], [0, 1, 1])


def test_load_annotations_unknown_labels_ignored(tmp_path):
    anno_path = tmp_path / 'a.json'
    anno_path.write_text(json.dumps({
        "image_path": "images/x.jpg",
        "labels": ["a1_age_approx", "a550_missing_attr"],
    }))
    list_path = tmp_path / 'list.txt'
    list_path.write_text('a.json\n')

    keys, label_rows, stats = load_annotations(str(list_path), ATTR_ID_TO_IDX,
                                               ds_root=str(tmp_path))
    assert stats['loaded'] == 1
    assert stats['unknown_attr_ids'] == {'a550_missing_attr': 1}
    np.testing.assert_allclose(label_rows[0], [0, 1, 0])


def test_load_annotations_empty_labels_falls_back_to_safe(tmp_path):
    anno_path = tmp_path / 'a.json'
    anno_path.write_text(json.dumps({"image_path": "images/x.jpg", "labels": []}))
    list_path = tmp_path / 'list.txt'
    list_path.write_text('a.json\n')

    keys, label_rows, stats = load_annotations(str(list_path), ATTR_ID_TO_IDX,
                                               ds_root=str(tmp_path))
    assert stats['loaded'] == 1
    np.testing.assert_allclose(label_rows[0], [1, 0, 0])  # safe only


def test_load_annotations_missing_json_skipped(tmp_path):
    list_path = tmp_path / 'list.txt'
    list_path.write_text('does_not_exist.json\n')
    keys, label_rows, stats = load_annotations(str(list_path), ATTR_ID_TO_IDX,
                                               ds_root=str(tmp_path))
    assert stats['loaded'] == 0
    assert keys == [] and label_rows == []


def test_write_and_validate_output_roundtrip(tmp_path):
    keys = ['2017_10001018.jpg', '2017_10035653.jpg']
    scores = np.arange(6, dtype=np.float32).reshape(2, 3)
    outfile = str(tmp_path / 'user_scores.tsv')

    write_scores_tsv(keys, scores, outfile)
    validate_output(outfile, expected_rows=2, n_users=3, logger=None)

    with open(outfile) as f:
        header = f.readline().strip().split('\t')
    assert header[0] == 'image_id'
    assert header[1:] == ['score_0', 'score_1', 'score_2']


def test_validate_output_rejects_bad_rows(tmp_path):
    outfile = tmp_path / 'bad.tsv'
    outfile.write_text('image_id\tscore_0\tscore_1\n'
                       'a.jpg\t1\t2\n'
                       'b.jpg\t1\tnot_a_number\n')
    with pytest.raises(RuntimeError, match='non-numeric'):
        validate_output(str(outfile), expected_rows=2, n_users=2, logger=None)


def test_validate_output_rejects_row_count_mismatch(tmp_path):
    outfile = tmp_path / 'short.tsv'
    outfile.write_text('image_id\tscore_0\nim.jpg\t1.0\n')
    with pytest.raises(RuntimeError, match='Row count mismatch'):
        validate_output(str(outfile), expected_rows=2, n_users=1, logger=None)


def test_validate_output_accepts_headerless_file(tmp_path):
    outfile = tmp_path / 'noheader.tsv'
    outfile.write_text('im1.jpg\t1.0\t2.0\nim2.jpg\t3.0\t4.0\n')
    # No header; first data row is detected (last cell numeric)
    validate_output(str(outfile), expected_rows=2, n_users=2, logger=None)


def test_end_to_end_via_main(tmp_path, monkeypatch):
    anno1 = _write_anno(tmp_path, 'images/train/xxxx.jpg',
                        ['a1_age_approx', 'a2_weight_approx'])
    anno2 = _write_anno(tmp_path, 'images/train/yyyy.jpg', ['a1_age_approx'])
    list_path = tmp_path / 'list.txt'
    list_path.write_text('\n'.join([os.path.basename(anno1),
                                    os.path.basename(anno2)]) + '\n')
    prefs = _write_user_prefs(tmp_path)
    outfile = str(tmp_path / 'out.tsv')

    monkeypatch.setattr(sys, 'argv', [
        'prepare_user_scores',
        '--anno-list', str(list_path),
        '--user-prefs', prefs,
        '--outfile', outfile,
        '--ds-root', str(tmp_path),
        '--pool', 'sum',
    ])
    # Avoid logging Rebinding issues in module: main() re-uses module logger
    from vispr.tools.scripts.prepare_user_scores import main
    main()

    assert os.path.exists(outfile)
    with open(outfile) as f:
        lines = [ln.rstrip('\n') for ln in f if ln.strip()]
    assert len(lines) == 3  # header + 2 rows
    header = lines[0].split('\t')
    assert header[0] == 'image_id' and len(header) == 3
    row = lines[1].split('\t')
    assert row[0] == 'xxxx.jpg'
    # sum pool: user0 = 1.0 + 3.0 = 4.0; user1 = 2.0 + 4.0 = 6.0
    assert abs(float(row[1]) - 4.0) < 1e-6
    assert abs(float(row[2]) - 6.0) < 1e-6