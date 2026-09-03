"""Run attribute predictions using a PyTorch model.

This script is a lightweight replacement for the original Caffe-based
`attribute_predict.py`. It uses a PyTorch model (e.g., torchvision backbone
with a final linear layer) to compute per-attribute probabilities.

Example usage:
    python attribute_predict_torch.py --arch resnet50 --weights my_model.pth \
        --infile test2017.txt --outfile preds.jsonl --batch-size 32 --num-classes 68
"""
import argparse
import os
import os.path as osp
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models

from vispr.datasets.pap_dataset import PAPDataset

try:
    from data.tar_streaming import StreamingConfig, StreamingPAPDataset
    STREAMING_AVAILABLE = True
except ImportError:
    STREAMING_AVAILABLE = False
from vispr.tools.common.logger import get_logger
logger = get_logger('inference')
# Route print to logger.info (scripts use print sparingly)
def _logger_print(*args, **kwargs):
    sep = kwargs.get('sep', ' ')
    end = kwargs.get('end', '\n')
    message = sep.join(map(str, args))
    logger.info(message)
print = _logger_print


def build_model(arch: str, num_classes: int, pretrained: bool = False):
    arch = arch.lower()
    if arch.startswith('resnet'):
        model = getattr(models, arch)(pretrained=pretrained)
        # Replace final fc
        in_f = model.fc.in_features
        model.fc = nn.Linear(in_f, num_classes)
        return model
    elif arch.startswith('mobilenet'):
        model = getattr(models, arch)(pretrained=pretrained)
        in_f = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_f, num_classes)
        return model
    else:
        raise ValueError('Unsupported arch: {}'.format(arch))


def classify_paths(model, device, loader, dataset, out_file: str):
    model = model.to(device)
    model.eval()

    with open(out_file, 'w') as wf:
        idx = 0
        for batch in loader:
            if len(batch) == 3:
                images, _, batch_paths = batch
            else:
                images, _ = batch
                batch_paths = None

            images = images.to(device).float()
            with torch.no_grad():
                outputs = model(images)
                probs = torch.sigmoid(outputs).cpu().numpy()

            batch_size_local = probs.shape[0]
            for b in range(batch_size_local):
                if batch_paths is not None:
                    ann_path = batch_paths[b]
                elif hasattr(dataset, 'anno_paths'):
                    ann_path = dataset.anno_paths[idx]
                else:
                    ann_path = None
                entry = {'anno_path': ann_path, 'pred_probs': probs[b].tolist()}
                wf.write(json.dumps(entry) + '\n')
                idx += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='resnet50')
    parser.add_argument('--weights', type=str, default=None, help='Path to model weights (.pth)')
    parser.add_argument('--infile', type=str, default=None, help='List of annotation paths to classify (local mode)')
    parser.add_argument('--outfile', type=str, required=True, help='Output JSONL file for predictions')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-classes', type=int, default=68)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained backbone')

    # Data source configuration for large tar streaming datasets
    parser.add_argument('--data-source', default='local',
                        choices=['local', 'hf_tar_stream', 'local_tar_stream'],
                        help='Data source: "local" for local files, "hf_tar_stream" for HuggingFace streaming, "local_tar_stream" for local tar.gz streaming')
    parser.add_argument('--config', default=None, help='Path to YAML config file (optional)')
    parser.add_argument('--hf-repo', default=None, help='HuggingFace repository ID (e.g., "username/dataset-name")')
    parser.add_argument('--hf-file-path', default=None, help='Path to combined .tar.gz file in HF repository')
    parser.add_argument('--hf-image-archive', default=None, help='Path to image .tar.gz file in HF repository')
    parser.add_argument('--hf-anno-archive', default=None, help='Path to annotation .tar.gz file in HF repository')
    parser.add_argument('--hf-anno-list', default=None, help='Path to .txt file listing annotations (in HF repo or local)')
    # Local tar streaming arguments
    parser.add_argument('--local-file-path', default=None, help='Path to a local combined .tar.gz archive')
    parser.add_argument('--local-image-archive', default=None, help='Path to a local image .tar.gz archive')
    parser.add_argument('--local-anno-archive', default=None, help='Path to a local annotation .tar.gz archive')
    parser.add_argument('--local-anno-list', default=None, help='Path to a local .txt file listing annotations to load')
    parser.add_argument('--buffer-size', type=int, default=1000, help='In-memory shuffle buffer size for streaming')
    parser.add_argument('--chunk-size', type=int, default=8*1024*1024, help='Read chunk size in bytes for tar streaming (default 8MB)')
    parser.add_argument('--cache-dir', default=None, help='Local directory for record-level caching (avoids re-streaming)')
    parser.add_argument('--log-interval', type=int, default=100, help='Log streaming progress every N records (0 to disable)')
    parser.add_argument('--max-retries', type=int, default=3, help='Max retries on network errors before giving up')
    args = parser.parse_args()

    config = None
    if args.config and os.path.exists(args.config):
        if not STREAMING_AVAILABLE:
            print("Warning: Config file provided but streaming module not available. Install with: pip install huggingface_hub PyYAML")
        else:
            config = StreamingConfig.from_yaml(args.config)
            if args.data_source:
                config.data_source = args.data_source
            if args.hf_repo:
                config.repo_id = args.hf_repo
            if args.hf_file_path:
                config.file_path = args.hf_file_path
            if args.hf_image_archive:
                config.image_archive_path = args.hf_image_archive
            if args.hf_anno_archive:
                config.annotation_archive_path = args.hf_anno_archive
            if args.hf_anno_list:
                config.anno_list_path = args.hf_anno_list
            # Local tar streaming overrides
            if args.local_file_path:
                config.file_path = args.local_file_path
            if args.local_image_archive:
                config.image_archive_path = args.local_image_archive
            if args.local_anno_archive:
                config.annotation_archive_path = args.local_anno_archive
            if args.local_anno_list:
                config.anno_list_path = args.local_anno_list
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

    data_source = args.data_source
    if config:
        data_source = config.data_source

    model = build_model(args.arch, args.num_classes, pretrained=args.pretrained)
    if args.weights is not None:
        ckpt = torch.load(args.weights, map_location='cpu')
        # Support either state_dict or full model checkpoint
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            state = ckpt['state_dict']
        else:
            state = ckpt
        # Some checkpoints have 'module.' prefixes from DataParallel
        new_state = {}
        for k, v in state.items():
            nk = k.replace('module.', '')
            new_state[nk] = v
        model.load_state_dict(new_state, strict=False)

    device = torch.device(args.device)

    if data_source == 'hf_tar_stream':
        if not STREAMING_AVAILABLE:
            raise ImportError(
                "Streaming module not available. Install dependencies with:\n"
                "pip install huggingface_hub PyYAML"
            )

        if config is None:
            if not args.hf_repo:
                raise ValueError("For HuggingFace streaming, --hf-repo is required")

            is_dual_mode = (args.hf_image_archive or args.hf_anno_archive or args.hf_anno_list)
            if is_dual_mode:
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

        dataset = StreamingPAPDataset(config=config, im_shape=(224, 224), shuffle=False, return_metadata=True)
        loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)

    elif data_source == 'local_tar_stream':
        if not STREAMING_AVAILABLE:
            raise ImportError(
                "Streaming module not available. Install dependencies with:\n"
                "pip install huggingface_hub PyYAML"
            )

        if config is None:
            is_dual_mode = (args.local_image_archive or args.local_anno_archive or args.local_anno_list)
            if is_dual_mode:
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
        if config.is_dual_archive_mode():
            print(f"Using local tar streaming (dual archive mode):")
            print(f"  Images: {config.image_archive_path}")
            print(f"  Annotations: {config.annotation_archive_path}")
            print(f"  Anno list: {config.anno_list_path}")
        else:
            print(f"Using local tar streaming (combined archive mode):")
            print(f"  File: {config.file_path}")

        dataset = StreamingPAPDataset(config=config, im_shape=(224, 224), shuffle=False, return_metadata=True)
        loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)

    else:
        if not args.infile:
            raise ValueError("For local data source, --infile is required")
        dataset = PAPDataset(args.infile, im_shape=(224, 224))
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    classify_paths(model, device, loader, dataset, args.outfile)


if __name__ == '__main__':
    main()

