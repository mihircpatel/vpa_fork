"""Unit tests for the PrivacyAwareAttributeModel (PRCNN extension).

Verifies the additional privacy-score layers defined in
``models/googlenet-prcnn/train_val.prototxt`` (lines 2121+):
    fc_ps_1 (Linear 68->128 + Sigmoid) -> fc_ps_2 (Linear 128->128 + Sigmoid)
    -> fc9 (Linear 128->30)

Also verifies dataset user-scores support and backward compatibility of
the base AttributeModel.
"""
import os
import json
import tempfile

import pytest
import torch
import torch.nn as nn

from vispr.models import build_model
from vispr.models.privacy_aware_model import PrivacyAwareAttributeModel as _PA
from vispr.models.attribute_model import AttributeModel as _Attr
from vispr.datasets.pap_dataset import PAPDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dummy_batch(batch_size=4, num_classes=68, num_scores=30, arch='resnet18'):
    """Random input + labels + user scores for a tiny backbone."""
    x = torch.randn(batch_size, 3, 224, 224)
    labels = torch.rand(batch_size, num_classes) > 0.5
    labels = labels.float()
    user_scores = torch.rand(batch_size, num_scores)
    return x, labels, user_scores


def _create_dummy_data(tmp_dir, n=2):
    """Create dummy images + annotation JSONs + list file + user scores TSV."""
    from PIL import Image
    import numpy as np

    img_dir = os.path.join(tmp_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)

    anno_paths = []
    for i in range(n):
        img_path = os.path.join(img_dir, f'img{i}.jpg')
        Image.fromarray((np.random.rand(64, 64, 3) * 255).astype('uint8')).save(img_path)
        anno = {'image_path': img_path, 'labels': ['a0_safe', 'a3_height_approx' if i == 1 else 'a2_weight_approx'], 'safe': True}
        anno_path = os.path.join(tmp_dir, f'anno{i}.json')
        with open(anno_path, 'w') as f:
            json.dump(anno, f)
        anno_paths.append(anno_path)

    list_path = os.path.join(tmp_dir, 'list.txt')
    with open(list_path, 'w') as f:
        f.write('\n'.join(anno_paths) + '\n')

    user_scores_path = os.path.join(tmp_dir, 'user_scores.tsv')
    with open(user_scores_path, 'w') as f:
        f.write('image_id\tscore_0\tscore_1\t...\tscore_29\n')
        f.write('img0.jpg\t' + '\t'.join([str(round(0.1 * (j % 5), 2)) for j in range(30)]) + '\n')
        f.write('img1.jpg\t' + '\t'.join([str(round(0.2 * (j % 5), 2)) for j in range(30)]) + '\n')

    return list_path, user_scores_path


# ---------------------------------------------------------------------------
# 1. Architecture
# ---------------------------------------------------------------------------

class TestArchitecture:
    def test_privacy_branch_exists(self):
        model = _PA(arch='resnet18', num_classes=68, num_privacy_scores=30)
        assert hasattr(model, 'privacy_branch')
        assert isinstance(model.privacy_branch, nn.Sequential)

    def test_privacy_branch_layer_shapes(self):
        model = _PA(arch='resnet18', num_classes=68, num_privacy_scores=30)
        layers = [m for m in model.privacy_branch if isinstance(m, nn.Linear)]
        assert len(layers) == 3
        assert layers[0].in_features == 68 and layers[0].out_features == 128   # fc_ps_1
        assert layers[1].in_features == 128 and layers[1].out_features == 128   # fc_ps_2
        assert layers[2].in_features == 128 and layers[2].out_features == 30    # fc9

    def test_state_dict_keys(self):
        model = _PA(arch='resnet18', num_classes=68, num_privacy_scores=30)
        keys = model.state_dict().keys()
        assert any('privacy_branch.0.weight' in k for k in keys)  # fc_ps_1
        assert any('privacy_branch.2.weight' in k for k in keys)  # fc_ps_2
        assert any('privacy_branch.4.weight' in k for k in keys)  # fc9

    def test_backbone_keys_identical_to_base(self):
        base = _Attr(arch='resnet18', num_classes=68)
        pa = _PA(arch='resnet18', num_classes=68, num_privacy_scores=30)
        base_keys = set(base.state_dict().keys())
        pa_keys = set(pa.state_dict().keys())
        assert base_keys <= pa_keys  # all base keys present in privacy-aware model


# ---------------------------------------------------------------------------
# 2. Forward pass output shapes
# ---------------------------------------------------------------------------

class TestForward:
    def test_returns_tuple(self):
        model = _PA(arch='resnet18', num_classes=68, num_privacy_scores=30)
        model.eval()
        x, _, _ = _dummy_batch()
        with torch.no_grad():
            out = model(x)
        assert isinstance(out, tuple)
        assert len(out) == 2

    def test_output_shapes(self):
        model = _PA(arch='resnet18', num_classes=68, num_privacy_scores=30)
        model.eval()
        x, _, _ = _dummy_batch()
        with torch.no_grad():
            attr_logits, privacy_scores = model(x)
        assert attr_logits.shape == (4, 68)
        assert privacy_scores.shape == (4, 30)

    def test_base_model_output_unchanged(self):
        model = _Attr(arch='resnet18', num_classes=68)
        model.eval()
        x, _, _ = _dummy_batch()
        with torch.no_grad():
            out = model(x)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (4, 68)


# ---------------------------------------------------------------------------
# 3. Backward pass / gradient flow
# ---------------------------------------------------------------------------

class TestBackward:
    def test_gradients_flow_to_privacy_branch(self):
        model = _PA(arch='resnet18', num_classes=68, num_privacy_scores=30)
        model.train()
        x, labels, user_scores = _dummy_batch()

        attr_logits, privacy_scores = model(x)
        loss = nn.BCEWithLogitsLoss()(attr_logits, labels) \
            + 0.03 * nn.MSELoss()(privacy_scores, user_scores)
        loss.backward()

        names = [n for n, p in model.named_parameters() if p.grad is not None]
        assert any('privacy_branch.0.weight' == n for n in names)
        assert any('privacy_branch.2.weight' == n for n in names)
        assert any('privacy_branch.4.weight' == n for n in names)
        assert any('backbone.fc.weight' == n for n in names)

    def test_privacy_loss_reduces(self):
        model = _PA(arch='resnet18', num_classes=68, num_privacy_scores=30)
        opt = torch.optim.SGD(model.parameters(), lr=0.05)
        x, labels, user_scores = _dummy_batch()

        losses = []
        for _ in range(3):
            opt.zero_grad()
            attr_logits, privacy_scores = model(x)
            loss = nn.BCEWithLogitsLoss()(attr_logits, labels) \
                + 0.03 * nn.MSELoss()(privacy_scores, user_scores)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        assert losses[-1] < losses[0]


# ---------------------------------------------------------------------------
# 4. Loading base weights into privacy-aware model
# ---------------------------------------------------------------------------

class TestWeightLoading:
    def test_load_base_checkpoint_strict_false(self):
        """strict=False loads backbone weights, keeps privacy branch random."""
        base = _Attr(arch='resnet18', num_classes=68)
        for p in base.parameters():
            p.data.uniform_(-0.5, 0.5)
        base_sd = {k: v.clone() for k, v in base.state_dict().items()}

        pa = _PA(arch='resnet18', num_classes=68, num_privacy_scores=30)
        result = pa.load_state_dict(base_sd, strict=False)
        # No unexpected keys (only missing privacy branch)
        assert len(result.unexpected_keys) == 0
        assert len(result.missing_keys) == 6  # 3 linear layers × (weight, bias)

        # Backbone weights fully matched
        for k, v in base_sd.items():
            assert torch.equal(pa.state_dict()[k], v)

    def test_privacy_branch_initialized_random(self):
        pa = _PA(arch='resnet18', num_classes=68, num_privacy_scores=30)
        w = pa.privacy_branch[4].weight
        assert not torch.equal(w, torch.zeros_like(w))


# ---------------------------------------------------------------------------
# 5. Training checkpoint roundtrip
# ---------------------------------------------------------------------------

class TestCheckpointRoundtrip:
    def test_save_and_load(self):
        model = _PA(arch='resnet18', num_classes=68, num_privacy_scores=30)
        for p in model.parameters():
            p.data.add_(0.01)
        original_sd = {k: v.clone() for k, v in model.state_dict().items()}

        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            path = f.name
        try:
            ckpt = {'epoch': 1, 'state_dict': model.state_dict(), 'optimizer': {}}
            torch.save(ckpt, path)

            model2 = _PA(arch='resnet18', num_classes=68, num_privacy_scores=30)
            from vispr.tools.common.utils import reload_model_weights
            reload_model_weights(model2, path, strict=True)
            for k in original_sd:
                assert torch.equal(model2.state_dict()[k], original_sd[k])
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# 6. Dataset user-scores support
# ---------------------------------------------------------------------------

class TestDatasetUserScores:
    def test_without_user_scores_returns_tuple_of_2(self):
        tmp = tempfile.mkdtemp()
        list_path, _ = _create_dummy_data(tmp, n=1)
        dataset = PAPDataset(list_path, im_shape=(64, 64))
        item = dataset[0]
        assert len(item) == 2

    def test_with_user_scores_returns_tuple_of_3(self):
        tmp = tempfile.mkdtemp()
        list_path, user_scores_path = _create_dummy_data(tmp, n=1)
        dataset = PAPDataset(list_path, im_shape=(64, 64), user_scores_path=user_scores_path)
        item = dataset[0]
        assert len(item) == 3
        img, labels, scores = item
        assert scores.shape[0] == 30

    def test_user_scores_lookup(self):
        tmp = tempfile.mkdtemp()
        list_path, user_scores_path = _create_dummy_data(tmp, n=1)
        dataset = PAPDataset(list_path, im_shape=(64, 64), user_scores_path=user_scores_path)
        _, _, scores = dataset[0]
        # img0.jpg has scores of round(0.1 * (j % 5), 2)
        assert scores.shape == (30,)
        assert scores[1].item() == pytest.approx(0.1)  # matches first data row
        assert (scores >= 0).all()


# ---------------------------------------------------------------------------
# 7. Backward compatibility - build_model
# ---------------------------------------------------------------------------

class TestBuildModel:
    def test_default_is_attribute(self):
        model = build_model('resnet18', 68)
        assert not hasattr(model, 'privacy_branch')

    def test_privacy_aware_build(self):
        model = build_model('resnet18', 68, model_type='privacy_aware', num_privacy_scores=30)
        assert hasattr(model, 'privacy_branch')
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert isinstance(out, tuple)
        assert out[0].shape == (1, 68)
        assert out[1].shape == (1, 30)

    def test_supported_archs(self):
        for arch in ('resnet18', 'resnet50'):
            model = build_model(arch, 68, model_type='privacy_aware')
            assert model is not None

    def test_unsupported_arch_raises(self):
        with pytest.raises(ValueError):
            build_model('not_a_real_arch', 68)


# ---------------------------------------------------------------------------
# 8. Training flow integration
# ---------------------------------------------------------------------------

class TestTrainingFlowIntegration:
    def test_train_script_build_model(self):
        from vispr.tools.scripts.train_torch import build_model as tb
        model = tb('resnet18', 68, model_type='privacy_aware', num_privacy_scores=30)
        assert hasattr(model, 'privacy_branch')

    def test_inference_script_build_model(self):
        from vispr.tools.scripts.attribute_predict_torch import build_model as ib
        model = ib('resnet18', 68, model_type='privacy_aware', num_privacy_scores=30)
        assert hasattr(model, 'privacy_branch')

    def test_train_one_epoch_privacy_aware(self):
        from vispr.tools.scripts.train_torch import train_one_epoch
        from torch.utils.data import DataLoader, TensorDataset

        model = _PA(arch='resnet18', num_classes=68, num_privacy_scores=30)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.BCEWithLogitsLoss()
        priv_criterion = nn.MSELoss()

        x, labels, user_scores = _dummy_batch(batch_size=8)
        ds = TensorDataset(x, labels, user_scores)
        loader = DataLoader(ds, batch_size=4, shuffle=False)

        loss = train_one_epoch(model, torch.device('cpu'), loader, opt, criterion,
                               epoch=1, privacy_criterion=priv_criterion,
                               privacy_loss_weight=0.03)
        assert loss > 0

    def test_validate_privacy_aware(self):
        from vispr.tools.scripts.train_torch import validate
        from torch.utils.data import DataLoader, TensorDataset

        model = _PA(arch='resnet18', num_classes=68, num_privacy_scores=30).eval()
        x, labels, user_scores = _dummy_batch(batch_size=8)
        ds = TensorDataset(x, labels, user_scores)
        loader = DataLoader(ds, batch_size=4, shuffle=False)

        mean_ap, ap_list = validate(model, torch.device('cpu'), loader)
        assert 0.0 <= mean_ap <= 1.0
        assert len(ap_list) == 68