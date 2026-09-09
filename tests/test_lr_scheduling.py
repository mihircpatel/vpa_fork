"""Tests for adaptive learning rate scheduling in train_torch.py.

Verifies:
  - Linear warmup from ~0 to target LR over warmup_epochs
  - ReduceLROnPlateau drops LR by 1/5 on loss stagnation
  - Checkpoint saves and restores scheduler state
  - Inference loads new-format checkpoints correctly
"""
import os
import tempfile
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau

from vispr.tools.common.utils import reload_model_weights


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model(num_classes=68):
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(16, num_classes),
    )


def _save(obj, path):
    torch.save(obj, path)
    return path


# ReduceLROnPlateau needs patience+2 step() calls to first reduce:
#   call 1: initialize best loss
#   calls 2..patience+1: count patience
#   call patience+2: reduce
PLATEAU_STEPS_TO_FIRST_REDUCE = 2  # extra steps beyond patience


# ---------------------------------------------------------------------------
# 1. LinearLR warmup schedule correctness
# ---------------------------------------------------------------------------

class TestLinearWarmup:
    def test_warmup_increases_lr_linearly(self):
        """LR should ramp from start_factor*lr to lr over warmup_epochs."""
        model = _make_model()
        base_lr = 1e-3
        warmup_epochs = 5
        optimizer = optim.Adam(model.parameters(), lr=base_lr)
        scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)

        lrs = []
        for _ in range(warmup_epochs):
            optimizer.step()
            scheduler.step()
            lrs.append(optimizer.param_groups[0]['lr'])

        # LR should increase each epoch during warmup
        for i in range(1, len(lrs)):
            assert lrs[i] > lrs[i - 1], f"LR did not increase at step {i}: {lrs[i-1]} -> {lrs[i]}"

        # Final LR should be close to base_lr
        assert lrs[-1] == pytest.approx(base_lr, rel=1e-3), \
            f"Final warmup LR {lrs[-1]} != base_lr {base_lr}"

    def test_warmup_initial_lr(self):
        """Before any step, LR should be start_factor * base_lr."""
        model = _make_model()
        base_lr = 1e-3
        start_factor = 0.01
        optimizer = optim.Adam(model.parameters(), lr=base_lr)
        LinearLR(optimizer, start_factor=start_factor, total_iters=5)

        lr_before = optimizer.param_groups[0]['lr']
        expected = base_lr * start_factor
        assert lr_before == pytest.approx(expected, rel=1e-5), \
            f"Initial LR {lr_before} != expected {expected}"


# ---------------------------------------------------------------------------
# 2. ReduceLROnPlateau drops LR on stagnation
# ---------------------------------------------------------------------------

class TestReduceOnPlateau:
    def test_lr_drops_on_plateau(self):
        """LR should reduce by factor when loss stagnates for patience epochs."""
        model = _make_model()
        base_lr = 1e-3
        patience = 3
        factor = 0.2
        optimizer = optim.Adam(model.parameters(), lr=base_lr)
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=factor, patience=patience
        )

        # ReduceLROnPlateau needs patience+2 step() calls to first reduce
        for _ in range(patience + PLATEAU_STEPS_TO_FIRST_REDUCE):
            optimizer.step()
            scheduler.step(1.0)

        lr_after = optimizer.param_groups[0]['lr']
        expected = base_lr * factor
        assert lr_after == pytest.approx(expected, rel=1e-3), \
            f"LR after plateau {lr_after} != expected {expected}"

    def test_lr_not_reduced_on_improvement(self):
        """LR should stay constant when loss keeps improving."""
        model = _make_model()
        base_lr = 1e-3
        optimizer = optim.Adam(model.parameters(), lr=base_lr)
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, patience=3
        )

        for epoch in range(10):
            optimizer.step()
            scheduler.step(1.0 - epoch * 0.1)

        lr_after = optimizer.param_groups[0]['lr']
        assert lr_after == pytest.approx(base_lr, rel=1e-3), \
            f"LR changed despite improving loss: {lr_after} != {base_lr}"

    def test_multiple_plateau_reductions(self):
        """LR should reduce multiple times if loss keeps stagnating."""
        model = _make_model()
        base_lr = 1e-3
        factor = 0.2
        patience = 2
        optimizer = optim.Adam(model.parameters(), lr=base_lr)
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=factor, patience=patience
        )

        # First plateau: patience+2 steps to reduce
        for _ in range(patience + PLATEAU_STEPS_TO_FIRST_REDUCE):
            optimizer.step()
            scheduler.step(1.0)
        lr1 = optimizer.param_groups[0]['lr']

        # Second plateau: another patience+1 steps (best_loss already set)
        for _ in range(patience + 1):
            optimizer.step()
            scheduler.step(1.0)
        lr2 = optimizer.param_groups[0]['lr']

        assert lr1 == pytest.approx(base_lr * factor, rel=1e-3)
        assert lr2 == pytest.approx(base_lr * factor * factor, rel=1e-3)

    def test_min_lr_floor(self):
        """LR should not go below min_lr."""
        model = _make_model()
        base_lr = 1e-3
        min_lr = 1e-5
        factor = 0.2
        optimizer = optim.Adam(model.parameters(), lr=base_lr)
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=factor, patience=1,
            min_lr=min_lr
        )

        for _ in range(20):
            for _ in range(3):  # patience=1 -> need 3 steps
                optimizer.step()
                scheduler.step(1.0)

        lr_final = optimizer.param_groups[0]['lr']
        assert lr_final >= min_lr, f"LR {lr_final} dropped below min_lr {min_lr}"


# ---------------------------------------------------------------------------
# 3. Combined warmup + plateau sequence
# ---------------------------------------------------------------------------

class TestCombinedSchedule:
    def test_warmup_then_plateau(self):
        """Full sequence: warmup for N epochs, then plateau-based reduction."""
        model = _make_model()
        base_lr = 1e-3
        warmup_epochs = 3
        patience = 2
        factor = 0.2
        optimizer = optim.Adam(model.parameters(), lr=base_lr)

        warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
        plateau_scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=factor, patience=patience
        )

        lrs = []
        total_epochs = 10
        for epoch in range(1, total_epochs + 1):
            optimizer.step()
            if epoch <= warmup_epochs:
                warmup_scheduler.step()
            else:
                plateau_scheduler.step(1.0)  # stagnant loss
            lrs.append(optimizer.param_groups[0]['lr'])

        # During warmup: LR should increase
        for i in range(1, warmup_epochs):
            assert lrs[i] > lrs[i - 1]

        # After warmup: LR should stay at base_lr until plateau triggers
        # Plateau needs patience+2 steps after warmup to reduce
        plateau_drop_idx = warmup_epochs + patience + 1  # 0-indexed
        if plateau_drop_idx < total_epochs:
            assert lrs[plateau_drop_idx] < lrs[plateau_drop_idx - 1]


# ---------------------------------------------------------------------------
# 4. Checkpoint roundtrip preserves scheduler state
# ---------------------------------------------------------------------------

class TestCheckpointSchedulerState:
    def test_checkpoint_saves_scheduler_state(self):
        """Checkpoint should contain warmup_scheduler and plateau_scheduler keys."""
        model = _make_model()
        base_lr = 1e-3
        optimizer = optim.Adam(model.parameters(), lr=base_lr)
        warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=5)
        plateau_scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, patience=3
        )

        for _ in range(3):
            optimizer.step()
            warmup_scheduler.step()
        for _ in range(2):
            optimizer.step()
            plateau_scheduler.step(1.0)

        ckpt = {
            'epoch': 5,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'warmup_scheduler': warmup_scheduler.state_dict(),
            'plateau_scheduler': plateau_scheduler.state_dict(),
        }

        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            path = f.name
        try:
            _save(ckpt, path)
            loaded = torch.load(path, weights_only=False)
            assert 'warmup_scheduler' in loaded, "Missing warmup_scheduler in checkpoint"
            assert 'plateau_scheduler' in loaded, "Missing plateau_scheduler in checkpoint"
        finally:
            os.unlink(path)

    def test_scheduler_state_restores_correctly(self):
        """Restoring scheduler state should resume from the same LR."""
        model = _make_model()
        base_lr = 1e-3
        optimizer = optim.Adam(model.parameters(), lr=base_lr)
        warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=5)
        plateau_scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, patience=3
        )

        for _ in range(3):
            optimizer.step()
            warmup_scheduler.step()
        lr_at_3 = optimizer.param_groups[0]['lr']

        ckpt = {
            'optimizer': optimizer.state_dict(),
            'warmup_scheduler': warmup_scheduler.state_dict(),
            'plateau_scheduler': plateau_scheduler.state_dict(),
        }

        model2 = _make_model()
        optimizer2 = optim.Adam(model2.parameters(), lr=base_lr)
        warmup2 = LinearLR(optimizer2, start_factor=0.01, total_iters=5)
        plateau2 = ReduceLROnPlateau(
            optimizer2, mode='min', factor=0.2, patience=3
        )

        optimizer2.load_state_dict(ckpt['optimizer'])
        warmup2.load_state_dict(ckpt['warmup_scheduler'])
        plateau2.load_state_dict(ckpt['plateau_scheduler'])

        lr_restored = optimizer2.param_groups[0]['lr']
        assert lr_restored == pytest.approx(lr_at_3, rel=1e-5), \
            f"Restored LR {lr_restored} != original {lr_at_3}"


# ---------------------------------------------------------------------------
# 5. Inference compatibility with new checkpoint format
# ---------------------------------------------------------------------------

class TestInferenceCompatibility:
    def test_inference_loads_new_checkpoint(self):
        """Inference should load new-format checkpoint (with scheduler keys) via reload_model_weights."""
        from vispr.tools.scripts.attribute_predict_torch import build_model

        num_classes = 68
        model = build_model('resnet18', num_classes, pretrained=False)
        for p in model.parameters():
            p.data.uniform_(-0.1, 0.1)
        original_sd = {k: v.clone() for k, v in model.state_dict().items()}

        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=5)
        plateau_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3)

        ckpt = {
            'epoch': 5,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'warmup_scheduler': warmup_scheduler.state_dict(),
            'plateau_scheduler': plateau_scheduler.state_dict(),
        }

        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            path = f.name
        try:
            _save(ckpt, path)
            model2 = build_model('resnet18', num_classes, pretrained=False)
            reload_model_weights(model2, path, strict=False, map_location='cpu')

            for k in original_sd:
                assert torch.equal(model2.state_dict()[k], original_sd[k]), \
                    f"Weight mismatch on key '{k}' after loading new checkpoint"
        finally:
            os.unlink(path)

    def test_train_checkpoint_loads_in_training(self):
        """Training checkpoint with scheduler state should be loadable."""
        model = _make_model()
        base_lr = 1e-3
        optimizer = optim.Adam(model.parameters(), lr=base_lr)
        warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=5)
        plateau_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3)

        for _ in range(3):
            optimizer.step()
            warmup_scheduler.step()
        optimizer.step()
        plateau_scheduler.step(1.0)

        ckpt = {
            'epoch': 4,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'warmup_scheduler': warmup_scheduler.state_dict(),
            'plateau_scheduler': plateau_scheduler.state_dict(),
        }

        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            path = f.name
        try:
            _save(ckpt, path)
            model2 = _make_model()
            optimizer2 = optim.Adam(model2.parameters(), lr=base_lr)
            warmup2 = LinearLR(optimizer2, start_factor=0.01, total_iters=5)
            plateau2 = ReduceLROnPlateau(optimizer2, mode='min', factor=0.2, patience=3)

            loaded = torch.load(path, weights_only=False)
            model2.load_state_dict(loaded['state_dict'])
            optimizer2.load_state_dict(loaded['optimizer'])
            warmup2.load_state_dict(loaded['warmup_scheduler'])
            plateau2.load_state_dict(loaded['plateau_scheduler'])

            lr_original = optimizer.param_groups[0]['lr']
            lr_restored = optimizer2.param_groups[0]['lr']
            assert lr_restored == pytest.approx(lr_original, rel=1e-5)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# 6. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_warmup_zero_epochs(self):
        """With warmup_epochs=0, plateau scheduler should activate immediately."""
        model = _make_model()
        base_lr = 1e-3
        optimizer = optim.Adam(model.parameters(), lr=base_lr)
        warmup_scheduler = LinearLR(optimizer, start_factor=1.0, total_iters=0)
        plateau_scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, patience=2
        )

        lrs = []
        for epoch in range(8):
            optimizer.step()
            if epoch == 0:
                warmup_scheduler.step()  # no-op for total_iters=0
            else:
                plateau_scheduler.step(1.0)  # stagnant
            lrs.append(optimizer.param_groups[0]['lr'])

        # patience=2 -> needs 4 step() calls to reduce (1 init + 2 patience + 1 reduce)
        # First call to plateau_scheduler is at epoch=1 (index 1 in lrs)
        # So reduction happens at epoch=1+3=4 (index 4 in lrs)
        assert lrs[3] == pytest.approx(base_lr, rel=1e-3)
        assert lrs[4] == pytest.approx(base_lr * 0.2, rel=1e-3)

    def test_high_start_factor(self):
        """start_factor=1.0 means no warmup (LR stays constant)."""
        model = _make_model()
        base_lr = 1e-3
        optimizer = optim.Adam(model.parameters(), lr=base_lr)
        scheduler = LinearLR(optimizer, start_factor=1.0, total_iters=5)

        lrs = []
        for _ in range(5):
            optimizer.step()
            scheduler.step()
            lrs.append(optimizer.param_groups[0]['lr'])

        for lr in lrs:
            assert lr == pytest.approx(base_lr, rel=1e-3)

    def test_plateau_with_cooldown(self):
        """Cooldown should delay next reduction after a drop."""
        model = _make_model()
        base_lr = 1e-3
        optimizer = optim.Adam(model.parameters(), lr=base_lr)
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, patience=1, cooldown=2
        )

        # patience=1 -> need 3 steps to first reduce
        for _ in range(3):
            optimizer.step()
            scheduler.step(1.0)
        lr_after_first = optimizer.param_groups[0]['lr']
        assert lr_after_first == pytest.approx(base_lr * 0.2, rel=1e-3)

        # During cooldown: 2 more stagnant steps should NOT reduce further
        for _ in range(2):
            optimizer.step()
            scheduler.step(1.0)
        lr_after_cooldown = optimizer.param_groups[0]['lr']
        assert lr_after_cooldown == pytest.approx(lr_after_first, rel=1e-3)
