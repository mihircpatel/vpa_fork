"""Simple training script for attribute prediction using PyTorch.

Trains a torchvision backbone with a linear head for multi-label
classification using BCEWithLogitsLoss.

Supports three data loading modes:
- 'local': loads from individual JSON annotation files (original behavior)
- 'hf_tar_stream': streams .tar.gz archives from Hugging Face Hub
- 'local_tar_stream': streams .tar.gz archives from local disk
"""
import argparse
import os
import json
import math
from pathlib import Path
import tempfile

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau
from sklearn.metrics import average_precision_score
import numpy as np

from vispr.datasets.pap_dataset import PAPDataset
from vispr.tools.common.utils import reload_model_weights
from vispr.tools.common.logger import get_logger
from vispr.models import build_model as build_model_factory
# Per-flow logger; writes to logs/train.log by default and mirrors to console.
logger = get_logger('train')
# Route existing print(...) calls to logger.info for minimal invasive changes
def _logger_print(*args, **kwargs):
    sep = kwargs.get('sep', ' ')
    end = kwargs.get('end', '\n')
    message = sep.join(map(str, args))
    logger.info(message)
print = _logger_print


def _atomic_torch_save(obj, target_path):
    """
    Atomically save `obj` to `target_path` using a temporary file and os.replace.
    Ensures parent directory exists. Safe for multithreaded concurrent saves (last-writer wins).
    """
    try:
        target_dir = os.path.dirname(target_path) or '.'
        # Safe directory creation even under concurrent access
        os.makedirs(target_dir, exist_ok=True)
        # Create a unique temporary file in same dir to ensure os.replace is atomic
        with tempfile.NamedTemporaryFile(delete=False, dir=target_dir, prefix='.tmp_save_', suffix='.pth') as tmpf:
            tmp_name = tmpf.name
        try:
            # Use torch.save to write to the temp file
            torch.save(obj, tmp_name)
            # Atomically move temp file to final destination
            os.replace(tmp_name, target_path)
        except Exception:
            # Cleanup temp file on error
            try:
                os.remove(tmp_name)
            except Exception:
                pass
            logger.exception('Failed to write temp checkpoint %s', tmp_name)
            raise
    except Exception as e:
        logger.error('Error saving checkpoint to %s: %s', target_path, e)
        raise


# Import streaming components (only if needed)
try:
    from data.tar_streaming import StreamingConfig, StreamingPAPDataset, get_streaming_dataloader
    STREAMING_AVAILABLE = True
except ImportError:
    STREAMING_AVAILABLE = False


def build_model(arch: str, num_classes: int, pretrained: bool = False,
                model_type: str = 'attribute', num_privacy_scores: int = 30):
    """Build a model by name.

    ``model_type`` selects between the base attribute model and the
    privacy-aware PRCNN extension.
    """
    return build_model_factory(
        arch=arch,
        num_classes=num_classes,
        model_type=model_type,
        num_privacy_scores=num_privacy_scores,
        pretrained=pretrained,
    )


def train_one_epoch(model, device, loader, optimizer, criterion, epoch, log_interval=50,
                    privacy_criterion=None, privacy_loss_weight=0.03):
    """Train one epoch.

    Supports both the base attribute model (returns ``attr_logits``) and the
    privacy-aware model (returns ``(attr_logits, privacy_scores)``).  When the
    model is privacy-aware, the total loss is::

        BCE(attr_logits, labels) + privacy_loss_weight * MSE(priv_scores, user_scores)
    """
    model.train()
    running_loss = 0.0
    running_attr_loss = 0.0
    running_priv_loss = 0.0
    running_correct = 0.0
    total_batches = 0
    for batch_idx, batch in enumerate(loader):
        if len(batch) == 3:
            data, target, user_scores = batch
        else:
            data, target = batch
            user_scores = None
        data = data.to(device).float()
        target = target.to(device).float()
        user_scores = user_scores.to(device).float() if user_scores is not None else None
        optimizer.zero_grad()
        outputs = model(data)

        if isinstance(outputs, tuple):
            attr_logits, privacy_scores = outputs
            attr_loss = criterion(attr_logits, target)
            if privacy_criterion is not None and user_scores is not None and privacy_scores is not None:
                privacy_loss = privacy_criterion(privacy_scores, user_scores)
            else:
                privacy_loss = torch.zeros((), device=device)
            loss = attr_loss + privacy_loss_weight * privacy_loss
            running_attr_loss += attr_loss.item()
            running_priv_loss += privacy_loss.item()
        else:
            loss = criterion(outputs, target)
            running_attr_loss += loss.item()

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total_batches += 1
        if batch_idx % log_interval == 0 and batch_idx > 0:
            # print(f'Epoch {epoch} [{batch_idx}/{len(loader)}] Loss: {running_loss / (batch_idx+1):.4f}')
            print(f'Epoch {epoch} [{batch_idx}/{total_batches}] Loss: {running_loss / (batch_idx + 1):.4f}')
    # train_loss = running_loss / len(loader)
    train_loss = running_loss / total_batches
    # train_acc = running_correct / len(loader)
    return train_loss


def validate(model, device, loader):
    model.eval()
    ys = []
    ys_pred = []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                data, target, _ = batch
            else:
                data, target = batch
            data = data.to(device).float()
            outputs = model(data)
            if isinstance(outputs, tuple):
                attr_logits, _ = outputs
            else:
                attr_logits = outputs
            probs = torch.sigmoid(attr_logits).cpu().numpy()
            ys_pred.append(probs)
            ys.append(target.numpy())
    ys = np.vstack(ys)
    ys_pred = np.vstack(ys_pred)
    # compute per-class average precision
    n_classes = ys.shape[1]
    ap_list = []
    for c in range(n_classes):
        try:
            ap = average_precision_score(ys[:, c], ys_pred[:, c])
        except Exception:
            ap = float('nan')
        ap_list.append(ap)
    mean_ap = np.nanmean(ap_list)
    return mean_ap, ap_list


def main():
    parser = argparse.ArgumentParser()
    # Data source configuration
    parser.add_argument('--data-source', default='local',
                        choices=['local', 'hf_tar_stream', 'local_tar_stream'],
                        help='Data source: "local" for local files, "hf_tar_stream" for HuggingFace streaming, "local_tar_stream" for local tar.gz streaming')
    parser.add_argument('--config', default=None, help='Path to YAML config file (optional)')

    # Local data arguments
    parser.add_argument('--infile', default=None, help='Annotation list (one JSON per line or path list)')
    parser.add_argument('--valfile', default=None, help='Validation annotation list (optional)')

    # HuggingFace streaming arguments - Combined archive mode
    parser.add_argument('--hf-repo', default=None, help='HuggingFace repository ID (e.g., "username/dataset-name")')
    parser.add_argument('--hf-file-path', default=None, help='Path to combined .tar.gz file in HF repository')
    parser.add_argument('--hf-val-file-path', default=None, help='Path to validation .tar.gz file in HF repository')

    # HuggingFace streaming arguments - Dual archive mode (separate image/annotation archives)
    parser.add_argument('--hf-image-archive', default=None, help='Path to image .tar.gz file in HF repository')
    parser.add_argument('--hf-anno-archive', default=None, help='Path to annotation .tar.gz file in HF repository')
    parser.add_argument('--hf-anno-list', default=None, help='Path to .txt file listing annotations (in HF repo or local)')
    parser.add_argument('--hf-val-image-archive', default=None, help='Path to validation image .tar.gz')
    parser.add_argument('--hf-val-anno-archive', default=None, help='Path to validation annotation .tar.gz')
    parser.add_argument('--hf-val-anno-list', default=None, help='Path to validation annotation list .txt')

    # Local tar streaming arguments - Combined archive mode
    parser.add_argument('--local-file-path', default=None, help='Path to a local combined .tar.gz archive')
    parser.add_argument('--local-val-file-path', default=None, help='Path to a local validation .tar.gz archive')

    # Local tar streaming arguments - Dual archive mode
    parser.add_argument('--local-image-archive', default=None, help='Path to a local image .tar.gz archive')
    parser.add_argument('--local-anno-archive', default=None, help='Path to a local annotation .tar.gz archive')
    parser.add_argument('--local-anno-list', default=None, help='Path to a local .txt file listing annotations to load')
    parser.add_argument('--local-val-image-archive', default=None, help='Path to a local validation image .tar.gz')
    parser.add_argument('--local-val-anno-archive', default=None, help='Path to a local validation annotation .tar.gz')
    parser.add_argument('--local-val-anno-list', default=None, help='Path to a local validation annotation list .txt')

    # Common streaming settings
    parser.add_argument('--buffer-size', type=int, default=1000, help='Shuffle buffer size for streaming')
    parser.add_argument('--chunk-size', type=int, default=8*1024*1024, help='Read chunk size in bytes for tar streaming (default 8MB)')
    parser.add_argument('--cache-dir', default=None, help='Local directory for record-level caching (avoids re-streaming)')
    parser.add_argument('--log-interval', type=int, default=100, help='Log streaming progress every N records (0 to disable)')
    parser.add_argument('--max-retries', type=int, default=3, help='Max retries on network errors before giving up')

    # Model and training arguments
    parser.add_argument('--arch', default='resnet50')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--model-type', default='attribute',
                        choices=['attribute', 'privacy_aware'],
                        help='Model class to use: "attribute" (base) or '
                             '"privacy_aware" (PRCNN extension with privacy score branch)')
    parser.add_argument('--num-privacy-scores', type=int, default=30,
                        help='Number of privacy score outputs for --model-type privacy_aware (default 30)')
    parser.add_argument('--user-scores-path', default=None,
                        help='TSV file with per-image user privacy scores for privacy-aware training '
                             '(train split)')
    parser.add_argument('--val-user-scores-path', default=None,
                        help='Independent TSV file with per-image user privacy scores for the '
                             'privacy-aware validation flow (default: none, validation does not '
                             'use privacy scores)')
    parser.add_argument('--privacy-loss-weight', type=float, default=0.03,
                        help='Weight of privacy score loss (matches prototxt loss_weight=0.03)')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num-classes', type=int, default=68)
    parser.add_argument('--save-path', default='model_final_best.pth')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')

    # Learning rate scheduling arguments
    parser.add_argument('--warmup-epochs', type=int, default=5,
                        help='Number of linear warmup epochs (LR ramps from ~0 to --lr)')
    parser.add_argument('--lr-patience', type=int, default=3,
                        help='Epochs to wait for loss improvement before reducing LR (ReduceLROnPlateau)')
    parser.add_argument('--lr-factor', type=float, default=0.2,
                        help='Factor to reduce LR by on plateau (0.2 = drop to 1/5)')
    parser.add_argument('--min-lr', type=float, default=1e-6,
                        help='Minimum learning rate floor')
    parser.add_argument('--cooldown', type=int, default=0,
                        help='Epochs to wait after a LR reduction before resuming patience counting')

    # Checkpoint download options
    parser.add_argument('--download-checkpoints', action='store_true', help='If set, download checkpoints before training')
    parser.add_argument('--checkpoint-provider', default='local', help='Provider for checkpoints: local, http, google_colab, s3, azure')
    parser.add_argument('--checkpoint-provider-config', default=None, help='JSON string or path to JSON file for provider config')
    parser.add_argument('--checkpoint-source', default='', help='Provider-specific source/folder (for local, a folder path)')
    parser.add_argument('--checkpoint-pattern', default='*.pth', help='Pattern to match checkpoint filenames')
    parser.add_argument('--checkpoint-dest', default=None, help='Local destination dir for downloaded checkpoints (defaults to save-path directory)')

    args = parser.parse_args()

    # Load configuration from YAML if provided
    config = None
    if args.config and os.path.exists(args.config):
        if not STREAMING_AVAILABLE:
            print("Warning: Config file provided but streaming module not available. Install with: pip install huggingface_hub PyYAML")
        else:
            config = StreamingConfig.from_yaml(args.config)
            # Override with command-line arguments
            if args.data_source:
                config.data_source = args.data_source
            if args.hf_repo:
                config.repo_id = args.hf_repo

            # Combined archive mode
            if args.hf_file_path:
                config.file_path = args.hf_file_path

            # Dual archive mode
            if args.hf_image_archive:
                config.image_archive_path = args.hf_image_archive
            if args.hf_anno_archive:
                config.annotation_archive_path = args.hf_anno_archive
            if args.hf_anno_list:
                config.anno_list_path = args.hf_anno_list

            # Local tar streaming - combined archive mode
            if args.local_file_path:
                config.file_path = args.local_file_path

            # Local tar streaming - dual archive mode
            if args.local_image_archive:
                config.image_archive_path = args.local_image_archive
            if args.local_anno_archive:
                config.annotation_archive_path = args.local_anno_archive
            if args.local_anno_list:
                config.anno_list_path = args.local_anno_list

            # Common settings
            if args.buffer_size:
                config.buffer_size = args.buffer_size
            if args.batch_size:
                config.batch_size = args.batch_size
            if args.num_classes:
                config.num_classes = args.num_classes
            if args.chunk_size:
                config.chunk_size = args.chunk_size
            if args.log_interval is not None:
                config.log_interval = args.log_interval
            if args.max_retries is not None:
                config.max_retries = args.max_retries
            if args.cache_dir:
                config.cache_dir = args.cache_dir

    # Determine data source
    data_source = args.data_source
    if config:
        data_source = config.data_source

    device = torch.device(args.device)

    # Create dataset and loader based on data source
    if data_source == 'hf_tar_stream':
        if not STREAMING_AVAILABLE:
            raise ImportError(
                "Streaming module not available. Install dependencies with:\n"
                "pip install huggingface_hub PyYAML"
            )

        # Create config from args if not loaded from file
        if config is None:
            if not args.hf_repo:
                raise ValueError("For HF streaming, --hf-repo is required")

            # Detect mode: dual archive or combined archive
            is_dual_mode = (args.hf_image_archive or args.hf_anno_archive or args.hf_anno_list)

            if is_dual_mode:
                # Dual archive mode
                if not all([args.hf_image_archive, args.hf_anno_archive, args.hf_anno_list]):
                    raise ValueError(
                        "For dual archive mode, --hf-image-archive, --hf-anno-archive, "
                        "and --hf-anno-list are all required"
                    )

                config = StreamingConfig(
                    data_source='hf_tar_stream',
                    repo_id=args.hf_repo,
                    image_archive_path=args.hf_image_archive,
                    annotation_archive_path=args.hf_anno_archive,
                    anno_list_path=args.hf_anno_list,
                    buffer_size=args.buffer_size,
                    batch_size=args.batch_size,
                    num_classes=args.num_classes,
                    chunk_size=args.chunk_size,
                    log_interval=args.log_interval,
                    max_retries=args.max_retries,
                    cache_dir=args.cache_dir,
                )
            else:
                # Combined archive mode
                if not args.hf_file_path:
                    raise ValueError("For combined archive mode, --hf-file-path is required")

                config = StreamingConfig(
                    data_source='hf_tar_stream',
                    repo_id=args.hf_repo,
                    file_path=args.hf_file_path,
                    buffer_size=args.buffer_size,
                    batch_size=args.batch_size,
                    num_classes=args.num_classes,
                    chunk_size=args.chunk_size,
                    log_interval=args.log_interval,
                    max_retries=args.max_retries,
                    cache_dir=args.cache_dir,
                )

        config.validate()

        # Print mode information
        if config.is_dual_archive_mode():
            print(f"Using HuggingFace streaming (dual archive mode):")
            print(f"  Repo: {config.repo_id}")
            print(f"  Images: {config.image_archive_path}")
            print(f"  Annotations: {config.annotation_archive_path}")
            print(f"  Anno list: {config.anno_list_path}")
        else:
            print(f"Using HuggingFace streaming (combined archive mode):")
            print(f"  Repo: {config.repo_id}")
            print(f"  File: {config.file_path}")

        dataset = StreamingPAPDataset(config=config, shuffle=True,
                                      user_scores_path=args.user_scores_path if args.model_type == 'privacy_aware' else None)
        # Note: IterableDataset doesn't support multiple workers well
        loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)
        print(f'Streaming dataset from HF Hub')

    elif data_source == 'local_tar_stream':
        if not STREAMING_AVAILABLE:
            raise ImportError(
                "Streaming module not available. Install dependencies with:\n"
                "pip install huggingface_hub PyYAML"
            )

        # Create config from args if not loaded from file
        if config is None:
            # Detect mode: dual archive or combined archive
            is_dual_mode = (args.local_image_archive or args.local_anno_archive or args.local_anno_list)

            if is_dual_mode:
                # Dual archive mode
                if not all([args.local_image_archive, args.local_anno_archive, args.local_anno_list]):
                    raise ValueError(
                        "For local dual archive mode, --local-image-archive, --local-anno-archive, "
                        "and --local-anno-list are all required"
                    )

                config = StreamingConfig(
                    data_source='local_tar_stream',
                    image_archive_path=args.local_image_archive,
                    annotation_archive_path=args.local_anno_archive,
                    anno_list_path=args.local_anno_list,
                    buffer_size=args.buffer_size,
                    batch_size=args.batch_size,
                    num_classes=args.num_classes,
                    chunk_size=args.chunk_size,
                    log_interval=args.log_interval,
                    max_retries=args.max_retries,
                    cache_dir=args.cache_dir,
                )
            else:
                # Combined archive mode
                if not args.local_file_path:
                    raise ValueError("For local combined archive mode, --local-file-path is required")

                config = StreamingConfig(
                    data_source='local_tar_stream',
                    file_path=args.local_file_path,
                    buffer_size=args.buffer_size,
                    batch_size=args.batch_size,
                    num_classes=args.num_classes,
                    chunk_size=args.chunk_size,
                    log_interval=args.log_interval,
                    max_retries=args.max_retries,
                    cache_dir=args.cache_dir,
                )

        config.validate()

        # Print mode information
        if config.is_dual_archive_mode():
            print(f"Using local tar streaming (dual archive mode):")
            print(f"  Images: {config.image_archive_path}")
            print(f"  Annotations: {config.annotation_archive_path}")
            print(f"  Anno list: {config.anno_list_path}")
        else:
            print(f"Using local tar streaming (combined archive mode):")
            print(f"  File: {config.file_path}")

        dataset = StreamingPAPDataset(config=config, shuffle=True,
                                      user_scores_path=args.user_scores_path if args.model_type == 'privacy_aware' else None)
        loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)
        print(f'Streaming dataset from local archives')

    else:
        # Local data loading (original behavior)
        if not args.infile:
            raise ValueError("For local data source, --infile is required")

        print(f"Using local data from: {args.infile}")
        user_scores_kwargs = {}
        if args.model_type == 'privacy_aware':
            user_scores_kwargs['user_scores_path'] = args.user_scores_path
            if args.user_scores_path is None:
                print("Warning: --model-type privacy_aware but no --user-scores-path provided. "
                      "Privacy loss will be skipped; attribute loss still applies.")
        dataset = PAPDataset(args.infile, im_shape=(224, 224), **user_scores_kwargs)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        print('No. of samples in train set: ' + str(len(loader.dataset)))

    # Setup validation loader
    if data_source == 'hf_tar_stream':
        if not STREAMING_AVAILABLE:
            print("Warning: Streaming not available, skipping validation")
        else:
            # Detect dual archive mode for validation
            has_dual_val = (args.hf_val_image_archive or args.hf_val_anno_archive or args.hf_val_anno_list)
            has_combined_val = args.hf_val_file_path

            if has_dual_val:
                # Dual archive mode validation
                if not all([args.hf_val_image_archive, args.hf_val_anno_archive, args.hf_val_anno_list]):
                    print("Warning: For dual archive validation, all of --hf-val-image-archive, "
                          "--hf-val-anno-archive, and --hf-val-anno-list are required. Skipping validation.")
                else:
                    val_config = StreamingConfig(
                        data_source='hf_tar_stream',
                        repo_id=config.repo_id if config else args.hf_repo,
                        image_archive_path=args.hf_val_image_archive,
                        annotation_archive_path=args.hf_val_anno_archive,
                        anno_list_path=args.hf_val_anno_list,
                        buffer_size=args.buffer_size,
                        batch_size=args.batch_size,
                        num_classes=args.num_classes,
                        chunk_size=args.chunk_size,
                        log_interval=args.log_interval,
                        max_retries=args.max_retries,
                        cache_dir=args.cache_dir,
                    )
                    val_dataset = StreamingPAPDataset(config=val_config, shuffle=False,
                                                      user_scores_path=args.val_user_scores_path if args.model_type == 'privacy_aware' else None)
                    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=0)
                    print(f'Streaming validation dataset from HF Hub (dual archive mode)')

            elif has_combined_val:
                # Combined archive mode validation
                val_config = StreamingConfig(
                    data_source='hf_tar_stream',
                    repo_id=config.repo_id if config else args.hf_repo,
                    file_path=args.hf_val_file_path,
                    buffer_size=args.buffer_size,
                    batch_size=args.batch_size,
                    num_classes=args.num_classes,
                    chunk_size=args.chunk_size,
                    log_interval=args.log_interval,
                    max_retries=args.max_retries,
                    cache_dir=args.cache_dir,
                )
                val_dataset = StreamingPAPDataset(config=val_config, shuffle=False,
                                                      user_scores_path=args.val_user_scores_path if args.model_type == 'privacy_aware' else None)
                val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=0)
                print(f'Streaming validation dataset from HF Hub (combined archive mode)')

    elif data_source == 'local_tar_stream':
        if not STREAMING_AVAILABLE:
            print("Warning: Streaming not available, skipping validation")
        else:
            # Detect dual archive mode for local validation
            has_dual_val = (args.local_val_image_archive or args.local_val_anno_archive or args.local_val_anno_list)
            has_combined_val = args.local_val_file_path

            if has_dual_val:
                # Dual archive mode validation
                if not all([args.local_val_image_archive, args.local_val_anno_archive, args.local_val_anno_list]):
                    print("Warning: For local dual archive validation, all of --local-val-image-archive, "
                          "--local-val-anno-archive, and --local-val-anno-list are required. Skipping validation.")
                else:
                    val_config = StreamingConfig(
                        data_source='local_tar_stream',
                        image_archive_path=args.local_val_image_archive,
                        annotation_archive_path=args.local_val_anno_archive,
                        anno_list_path=args.local_val_anno_list,
                        buffer_size=args.buffer_size,
                        batch_size=args.batch_size,
                        num_classes=args.num_classes,
                        chunk_size=args.chunk_size,
                        log_interval=args.log_interval,
                        max_retries=args.max_retries,
                        cache_dir=args.cache_dir,
                    )
                    val_dataset = StreamingPAPDataset(config=val_config, shuffle=False,
                                                      user_scores_path=args.val_user_scores_path if args.model_type == 'privacy_aware' else None)
                    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=0)
                    print(f'Streaming validation dataset from local archives (dual archive mode)')

            elif has_combined_val:
                # Combined archive mode validation
                val_config = StreamingConfig(
                    data_source='local_tar_stream',
                    file_path=args.local_val_file_path,
                    buffer_size=args.buffer_size,
                    batch_size=args.batch_size,
                    num_classes=args.num_classes,
                    chunk_size=args.chunk_size,
                    log_interval=args.log_interval,
                    max_retries=args.max_retries,
                    cache_dir=args.cache_dir,
                )
                val_dataset = StreamingPAPDataset(config=val_config, shuffle=False,
                                                      user_scores_path=args.val_user_scores_path if args.model_type == 'privacy_aware' else None)
                val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=0)
                print(f'Streaming validation dataset from local archives (combined archive mode)')

    elif data_source == 'local' and args.valfile:
        user_scores_kwargs = {}
        if args.model_type == 'privacy_aware' and args.val_user_scores_path:
            user_scores_kwargs['user_scores_path'] = args.val_user_scores_path
        val_dataset = PAPDataset(args.valfile, im_shape=(224, 224), **user_scores_kwargs)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
        print('No. of samples in validation set: ' + str(len(val_loader.dataset)))

    # Optionally download checkpoints before training
    dest_dir = None
    if args.download_checkpoints:
        # parse provider config
        pconfig = {}
        if args.checkpoint_provider_config:
            if os.path.isfile(args.checkpoint_provider_config):
                try:
                    with open(args.checkpoint_provider_config, 'r') as f:
                        pconfig = json.load(f)
                except Exception as e:
                    raise ValueError(f'Failed to load checkpoint provider config file: {e}')
            else:
                try:
                    pconfig = json.loads(args.checkpoint_provider_config)
                except Exception as e:
                    raise ValueError(f'Failed to parse checkpoint provider config: {e}')
        # destination dir defaults to directory of save_path
        if args.checkpoint_dest:
            dest_dir = args.checkpoint_dest
        else:
            # dest_dir = os.path.dirname(args.save_path) or '.'
            dest_dir = './checkpoints/downloaded'
        try:
            from vispr.tools.common.file_downloader import DownloadManager
            dm = DownloadManager(provider=args.checkpoint_provider, provider_config=pconfig, logger_obj=logger)
            downloaded = dm.download_checkpoints(folder=args.checkpoint_source, dest_dir=dest_dir, pattern=args.checkpoint_pattern)
            logger.info('Downloaded %d checkpoint(s) to %s', len(downloaded), dest_dir)
        except Exception as e:
            logger.error('Checkpoint download failed: %s', e)
            # proceed without failing training; user can choose to abort by removing flag

    if args.model_type == 'privacy_aware' and args.user_scores_path:
        print(f'Using privacy-aware model ({args.model_type}) '
              f'with train user scores from: {args.user_scores_path}')
        if args.val_user_scores_path:
            print(f'  Validation user scores from: {args.val_user_scores_path}')
        elif args.valfile:
            print(f'  Warning: no --val-user-scores-path provided; validation dataset will not '
                  f'load user scores.')
    model = build_model(args.arch, args.num_classes, pretrained=args.pretrained,
                        model_type=args.model_type,
                        num_privacy_scores=args.num_privacy_scores).to(device)
    if dest_dir is not None:
        model_weight_file_basename = os.path.basename(args.save_path)
        model_weight_file_path = os.path.join(dest_dir, model_weight_file_basename)
        if os.path.isfile(model_weight_file_path):
            print(f'Loading model weights from {model_weight_file_path}')
            reload_model_weights(model, model_weight_file_path, strict=False, map_location=device)
        else:
            print(f'No model weights found at {model_weight_file_path}, proceeding with random initialization')
    else:
        print(f'No checkpoint path {dest_dir} exists, proceeding with random initialization')

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()
    privacy_criterion = nn.MSELoss() if args.model_type == 'privacy_aware' else None

    # Learning rate scheduling: linear warmup then ReduceLROnPlateau
    warmup_scheduler = LinearLR(
        optimizer, start_factor=1e-2, total_iters=args.warmup_epochs
    )
    plateau_scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=args.lr_factor,
        patience=args.lr_patience, cooldown=args.cooldown,
        min_lr=args.min_lr
    )

    best_loss = math.inf
    val_loader = None

    for epoch in range(1, args.epochs + 1):
        avg_loss = train_one_epoch(model, device, loader, optimizer, criterion, epoch,
                                   privacy_criterion=privacy_criterion,
                                   privacy_loss_weight=args.privacy_loss_weight)
        print(f'Epoch {epoch} finished. Avg Loss: {avg_loss:.4f}')

        # Step LR schedulers
        if epoch <= args.warmup_epochs:
            warmup_scheduler.step()
        else:
            # Use validation loss if available, otherwise training loss
            plateau_metric = avg_loss
            plateau_scheduler.step(plateau_metric)
        print(f'  Current LR: {optimizer.param_groups[0]["lr"]:.2e}')

        # Save checkpoint each epoch (atomic, thread-safe)
        ckpt = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'warmup_scheduler': warmup_scheduler.state_dict(),
            'plateau_scheduler': plateau_scheduler.state_dict(),
        }
        try:
            _atomic_torch_save(ckpt, args.save_path)
        except Exception:
            print(f'Failed to save checkpoint to {args.save_path}')
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.splitext(args.save_path)[0] + '_best.pth'
            try:
                _atomic_torch_save(ckpt, best_path)
                print('Saved best model to', best_path)
            except Exception:
                print(f'Failed to save best model to {best_path}')

        if val_loader is not None:
            mean_ap, ap_list = validate(model, device, val_loader)
            print(f'Validation mAP: {mean_ap:.4f}')
            # Use validation loss for plateau scheduler if available
            # Note: validate() returns mAP, not loss. We use training loss as proxy
            # since we don't have a validation loss computed here.


if __name__ == '__main__':
    main()

