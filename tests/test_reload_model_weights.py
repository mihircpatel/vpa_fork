"""Tests for reload_model_weights in vispr.tools.common.utils.

Covers checkpoint format handling, prefix stripping, strict mode,
and integration with training and inference build_model functions.
"""
import os
import tempfile
import pytest
import torch
import torch.nn as nn

from vispr.tools.common.utils import reload_model_weights


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model(num_classes=68):
    """Create a tiny ResNet-like model for testing (no pretrained weights)."""
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(16, num_classes),
    )
    return model


def _state_dict(model):
    return {k: v.clone() for k, v in model.state_dict().items()}


def _save(obj, path):
    torch.save(obj, path)
    return path


# ---------------------------------------------------------------------------
# 1. None path
# ---------------------------------------------------------------------------

class TestNonePath:
    def test_returns_model_unchanged(self):
        model = _make_model()
        original_sd = _state_dict(model)
        result = reload_model_weights(model, None)
        assert result is model
        for k in model.state_dict():
            assert torch.equal(model.state_dict()[k], original_sd[k])


# ---------------------------------------------------------------------------
# 2. Raw state_dict (dict of param_name -> tensor)
# ---------------------------------------------------------------------------

class TestRawStateDict:
    def test_loads_raw_state_dict(self):
        model_a = _make_model()
        model_b = _make_model()

        sd = _state_dict(model_a)
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            path = f.name
        try:
            _save(sd, path)
            reload_model_weights(model_b, path, strict=True)
            for k in model_a.state_dict():
                assert torch.equal(model_a.state_dict()[k], model_b.state_dict()[k])
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# 3. Checkpoint dict with 'state_dict' key
# ---------------------------------------------------------------------------

class TestStateDictKey:
    def test_extracts_from_state_dict_key(self):
        model_a = _make_model()
        model_b = _make_model()

        ckpt = {'epoch': 5, 'state_dict': _state_dict(model_a), 'optimizer': {}}
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            path = f.name
        try:
            _save(ckpt, path)
            reload_model_weights(model_b, path, strict=True)
            for k in model_a.state_dict():
                assert torch.equal(model_a.state_dict()[k], model_b.state_dict()[k])
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# 4. Checkpoint dict with 'model_state_dict' key
# ---------------------------------------------------------------------------

class TestModelStateDictKey:
    def test_extracts_from_model_state_dict_key(self):
        model_a = _make_model()
        model_b = _make_model()

        ckpt = {'model_state_dict': _state_dict(model_a)}
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            path = f.name
        try:
            _save(ckpt, path)
            reload_model_weights(model_b, path, strict=True)
            for k in model_a.state_dict():
                assert torch.equal(model_a.state_dict()[k], model_b.state_dict()[k])
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# 5. Checkpoint dict with 'model' key
# ---------------------------------------------------------------------------

class TestModelKey:
    def test_extracts_from_model_key(self):
        model_a = _make_model()
        model_b = _make_model()

        ckpt = {'model': _state_dict(model_a)}
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            path = f.name
        try:
            _save(ckpt, path)
            reload_model_weights(model_b, path, strict=True)
            for k in model_a.state_dict():
                assert torch.equal(model_a.state_dict()[k], model_b.state_dict()[k])
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# 6. Prefix stripping: 'module.'
# ---------------------------------------------------------------------------

class TestModulePrefix:
    def test_strips_module_prefix(self):
        model_a = _make_model()
        model_b = _make_model()

        sd = _state_dict(model_a)
        prefixed_sd = {f'module.{k}': v for k, v in sd.items()}
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            path = f.name
        try:
            _save(prefixed_sd, path)
            reload_model_weights(model_b, path, strict=True)
            for k in model_a.state_dict():
                assert torch.equal(model_a.state_dict()[k], model_b.state_dict()[k])
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# 7. Prefix stripping: 'model.'
# ---------------------------------------------------------------------------

class TestModelPrefix:
    def test_strips_model_prefix(self):
        model_a = _make_model()
        model_b = _make_model()

        sd = _state_dict(model_a)
        prefixed_sd = {f'model.{k}': v for k, v in sd.items()}
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            path = f.name
        try:
            _save(prefixed_sd, path)
            reload_model_weights(model_b, path, strict=True)
            for k in model_a.state_dict():
                assert torch.equal(model_a.state_dict()[k], model_b.state_dict()[k])
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# 8. Combined prefix: 'module.model.'
# ---------------------------------------------------------------------------

class TestModuleModelPrefix:
    def test_strips_module_model_prefix(self):
        model_a = _make_model()
        model_b = _make_model()

        sd = _state_dict(model_a)
        prefixed_sd = {f'module.model.{k}': v for k, v in sd.items()}
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            path = f.name
        try:
            _save(prefixed_sd, path)
            reload_model_weights(model_b, path, strict=True)
            for k in model_a.state_dict():
                assert torch.equal(model_a.state_dict()[k], model_b.state_dict()[k])
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# 9. strict=False with mismatched keys
# ---------------------------------------------------------------------------

class TestStrictFalse:
    def test_loads_matching_keys_ignores_missing(self):
        model_a = _make_model(num_classes=68)
        model_b = _make_model(num_classes=68)

        # Add an extra key to model_b that won't be in checkpoint
        extra = nn.Linear(16, 10)
        model_b.extra = extra

        sd = _state_dict(model_a)
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            path = f.name
        try:
            _save(sd, path)
            # strict=False should succeed even though model_b has extra keys
            reload_model_weights(model_b, path, strict=False)
            for k in model_a.state_dict():
                assert torch.equal(model_a.state_dict()[k], model_b.state_dict()[k])
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# 10. strict=True with mismatched keys raises
# ---------------------------------------------------------------------------

class TestStrictTrue:
    def test_raises_on_unexpected_keys(self):
        """strict=True raises when checkpoint has keys the model doesn't expect."""
        model_a = _make_model(num_classes=68)
        model_b = _make_model(num_classes=68)

        sd = _state_dict(model_a)
        # Inject an extra key that model_b doesn't have
        sd['extra_layer.weight'] = torch.randn(10, 16)
        sd['extra_layer.bias'] = torch.randn(10)

        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            path = f.name
        try:
            _save(sd, path)
            with pytest.raises(RuntimeError, match="Unexpected key"):
                reload_model_weights(model_b, path, strict=True)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# 11. Invalid file path raises FileNotFoundError
# ---------------------------------------------------------------------------

class TestInvalidPath:
    def test_raises_on_missing_file(self):
        model = _make_model()
        with pytest.raises(FileNotFoundError):
            reload_model_weights(model, '/nonexistent/path/weights.pth')


# ---------------------------------------------------------------------------
# 12. Non-dict checkpoint raises TypeError
# ---------------------------------------------------------------------------

class TestNonDictCheckpoint:
    def test_raises_on_non_dict(self):
        model = _make_model()
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            path = f.name
        try:
            _save([1, 2, 3], path)
            with pytest.raises(TypeError, match="Expected a state_dict-like"):
                reload_model_weights(model, path)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# 13. Non-string keys raise TypeError
# ---------------------------------------------------------------------------

class TestNonStringKeys:
    def test_raises_on_non_string_keys(self):
        model = _make_model()
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            path = f.name
        try:
            _save({0: torch.randn(3), 1: torch.randn(16)}, path)
            with pytest.raises(TypeError, match="non-string key"):
                reload_model_weights(model, path)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# 14. Training flow integration: build_model → save checkpoint → reload → verify
# ---------------------------------------------------------------------------

class TestTrainingFlowIntegration:
    def test_checkpoint_roundtrip_matches(self):
        """Simulate the train_torch.py checkpoint save/load cycle."""
        from vispr.tools.scripts.train_torch import build_model

        num_classes = 68
        model = build_model('resnet18', num_classes, pretrained=False)

        # Simulate training: modify weights
        for p in model.parameters():
            p.data.add_(0.01)

        original_sd = {k: v.clone() for k, v in model.state_dict().items()}

        # Save as full checkpoint (like train_torch.py does)
        ckpt = {'epoch': 1, 'state_dict': model.state_dict(), 'optimizer': {}}
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            path = f.name
        try:
            _save(ckpt, path)

            # Reload into a fresh model
            model2 = build_model('resnet18', num_classes, pretrained=False)
            reload_model_weights(model2, path, strict=True)

            for k in original_sd:
                assert torch.equal(model2.state_dict()[k], original_sd[k]), \
                    f"Mismatch on key '{k}' after training checkpoint roundtrip"
        finally:
            os.unlink(path)

    def test_best_checkpoint_roundtrip_matches(self):
        """Simulate saving best checkpoint with different filename."""
        from vispr.tools.scripts.train_torch import build_model

        num_classes = 68
        model = build_model('resnet18', num_classes, pretrained=False)
        for p in model.parameters():
            p.data.mul_(0.5)

        original_sd = {k: v.clone() for k, v in model.state_dict().items()}

        ckpt = {'epoch': 3, 'state_dict': model.state_dict(), 'optimizer': {}}
        with tempfile.NamedTemporaryFile(suffix='_best.pth', delete=False) as f:
            path = f.name
        try:
            _save(ckpt, path)

            model2 = build_model('resnet18', num_classes, pretrained=False)
            reload_model_weights(model2, path, strict=True)

            for k in original_sd:
                assert torch.equal(model2.state_dict()[k], original_sd[k])
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# 15. Inference flow integration: build_model → save weights → reload → verify
# ---------------------------------------------------------------------------

class TestInferenceFlowIntegration:
    def test_raw_weights_roundtrip_matches(self):
        """Simulate the attribute_predict_torch.py weights load cycle."""
        from vispr.tools.scripts.attribute_predict_torch import build_model

        num_classes = 68
        model = build_model('resnet18', num_classes, pretrained=False)
        for p in model.parameters():
            p.data.add_(0.02)

        original_sd = {k: v.clone() for k, v in model.state_dict().items()}

        # Inference script saves as raw state_dict or full checkpoint
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            path = f.name
        try:
            _save(model.state_dict(), path)

            model2 = build_model('resnet18', num_classes, pretrained=False)
            reload_model_weights(model2, path, strict=False, map_location='cpu')

            for k in original_sd:
                assert torch.equal(model2.state_dict()[k], original_sd[k]), \
                    f"Mismatch on key '{k}' after inference weights roundtrip"
        finally:
            os.unlink(path)

    def test_full_checkpoint_roundtrip_matches(self):
        """Inference loading a full checkpoint (not raw state_dict)."""
        from vispr.tools.scripts.attribute_predict_torch import build_model

        num_classes = 68
        model = build_model('resnet18', num_classes, pretrained=False)
        for p in model.parameters():
            p.data.fill_(0.42)

        original_sd = {k: v.clone() for k, v in model.state_dict().items()}

        ckpt = {'epoch': 10, 'state_dict': model.state_dict(), 'optimizer': {}}
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            path = f.name
        try:
            _save(ckpt, path)

            model2 = build_model('resnet18', num_classes, pretrained=False)
            reload_model_weights(model2, path, strict=False, map_location='cpu')

            for k in original_sd:
                assert torch.equal(model2.state_dict()[k], original_sd[k])
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# 16. Cross-script compatibility: train saves → inference loads
# ---------------------------------------------------------------------------

class TestCrossScriptCompatibility:
    def test_train_checkpoint_loads_in_inference(self):
        """Verify a checkpoint saved by train_torch.py can be loaded by inference."""
        from vispr.tools.scripts.train_torch import build_model as train_build
        from vispr.tools.scripts.attribute_predict_torch import build_model as infer_build

        num_classes = 68
        model = train_build('resnet18', num_classes, pretrained=False)
        for p in model.parameters():
            p.data.uniform_(-0.1, 0.1)

        original_sd = {k: v.clone() for k, v in model.state_dict().items()}

        # Save as train_torch.py would (full checkpoint)
        ckpt = {'epoch': 1, 'state_dict': model.state_dict(), 'optimizer': {}}
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            path = f.name
        try:
            _save(ckpt, path)

            # Load with inference build_model
            model2 = infer_build('resnet18', num_classes, pretrained=False)
            reload_model_weights(model2, path, strict=False, map_location='cpu')

            for k in original_sd:
                assert torch.equal(model2.state_dict()[k], original_sd[k]), \
                    f"Cross-script mismatch on key '{k}'"
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# 17. map_location variants
# ---------------------------------------------------------------------------

class TestMapLocation:
    def test_map_location_cpu(self):
        model_a = _make_model()
        model_b = _make_model()
        sd = _state_dict(model_a)
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            path = f.name
        try:
            _save(sd, path)
            reload_model_weights(model_b, path, strict=True, map_location='cpu')
            for k in model_a.state_dict():
                assert model_b.state_dict()[k].device == torch.device('cpu')
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# 18. load_model_weights alias
# ---------------------------------------------------------------------------

class TestLoadModelWeightsAlias:
    def test_alias_works(self):
        from vispr.tools.common.utils import load_model_weights

        model_a = _make_model()
        model_b = _make_model()
        sd = _state_dict(model_a)
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            path = f.name
        try:
            _save(sd, path)
            load_model_weights(model_b, path, strict=True)
            for k in model_a.state_dict():
                assert torch.equal(model_a.state_dict()[k], model_b.state_dict()[k])
        finally:
            os.unlink(path)
