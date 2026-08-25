"""Example usage of HF Tar Streaming module.

This file demonstrates various ways to use the streaming data loader,
including caching, error handling, and integration with training.
"""

import torch
from torch.utils.data import DataLoader


# Example 1: Basic streaming usage
def example_basic_streaming():
    """Stream data from HuggingFace Hub - basic usage."""
    from data.tar_streaming import StreamingConfig, StreamingPAPDataset

    config = StreamingConfig(
        data_source='hf_tar_stream',
        repo_id='username/vispr-dataset',
        file_path='train.tar.gz',
        buffer_size=1000,
        batch_size=32
    )
    config.validate()

    dataset = StreamingPAPDataset(config=config, shuffle=True)
    loader = DataLoader(dataset, batch_size=32, num_workers=0)

    for batch_idx, (images, labels) in enumerate(loader):
        print(f"Batch {batch_idx}: images shape {images.shape}, labels shape {labels.shape}")
        if batch_idx >= 2:
            break


# Example 2: Using YAML configuration
def example_yaml_config():
    """Load configuration from YAML file."""
    from data.tar_streaming import StreamingConfig, StreamingPAPDataset

    config = StreamingConfig.from_yaml('configs/data_config.yaml')
    config.batch_size = 16
    config.buffer_size = 2000

    dataset = StreamingPAPDataset(config=config, shuffle=True)
    loader = DataLoader(dataset, batch_size=config.batch_size)
    return loader


# Example 3: Convenience wrapper
def example_convenience_wrapper():
    """Use the convenience wrapper for quick setup."""
    from data.tar_streaming import StreamingPAPDatasetFromFile

    dataset = StreamingPAPDatasetFromFile(
        repo_id='username/dataset',
        file_path='train.tar.gz',
        im_shape=(224, 224),
        buffer_size=1500,
        shuffle=True,
        mean=(104.0, 117.0, 123.0)
    )

    loader = DataLoader(dataset, batch_size=32)
    return loader


# Example 4: Direct streamer usage (advanced)
def example_direct_streamer():
    """Use HFTarStreamer directly for custom processing."""
    from data.tar_streaming import HFTarStreamer

    streamer = HFTarStreamer(
        repo_id='username/dataset',
        file_path='train.tar.gz'
    )

    for i, record in enumerate(streamer.extract_structured_data()):
        print(f"Record {i}:")
        print(f"  Image path: {record['image_path']}")
        print(f"  Image size: {record['image'].size}")
        print(f"  Labels: {record.get('labels', [])}")
        if i >= 5:
            break

    print(f"Streamer stats: {streamer.stats()}")


# Example 5: Training loop integration
def example_training_loop():
    """Example of integrating with a training loop."""
    import torch.nn as nn
    import torch.optim as optim
    from data.tar_streaming import StreamingConfig, StreamingPAPDataset

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = StreamingConfig(
        data_source='hf_tar_stream',
        repo_id='username/dataset',
        file_path='train.tar.gz',
        buffer_size=1000,
        batch_size=32
    )

    dataset = StreamingPAPDataset(config=config, shuffle=True)
    loader = DataLoader(dataset, batch_size=32)

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 224 * 224, 68)
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(2):
        running_loss = 0.0

        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(device).float()
            labels = labels.to(device).float()

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')

            if batch_idx >= 20:
                break

        print(f'Epoch {epoch} finished. Avg Loss: {running_loss / (batch_idx + 1):.4f}')

    # Show iteration stats
    print(f"Dataset stats: {dataset.stats()}")


# Example 6: Memory-efficient validation
def example_validation():
    """Example of validation with streaming data."""
    import numpy as np
    from sklearn.metrics import average_precision_score
    from data.tar_streaming import StreamingConfig, StreamingPAPDataset

    config = StreamingConfig(
        data_source='hf_tar_stream',
        repo_id='username/dataset',
        file_path='val.tar.gz',
        batch_size=32
    )

    dataset = StreamingPAPDataset(config=config, shuffle=False)
    loader = DataLoader(dataset, batch_size=32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(3 * 224 * 224, 68)
    ).to(device)

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device).float()
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()

            all_preds.append(probs)
            all_labels.append(labels.numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    aps = []
    for i in range(all_labels.shape[1]):
        try:
            ap = average_precision_score(all_labels[:, i], all_preds[:, i])
            aps.append(ap)
        except Exception:
            aps.append(0.0)

    mean_ap = np.mean(aps)
    print(f'Validation mAP: {mean_ap:.4f}')


# Example 7: Command-line argument parsing
def example_cli_integration():
    """Example of integrating with command-line arguments."""
    import argparse
    from data.tar_streaming import StreamingConfig

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-source', default='local')
    parser.add_argument('--hf-repo', default=None)
    parser.add_argument('--hf-file-path', default=None)
    parser.add_argument('--buffer-size', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--chunk-size', type=int, default=8*1024*1024)
    parser.add_argument('--cache-dir', default=None)
    parser.add_argument('--max-retries', type=int, default=3)
    parser.add_argument('--log-interval', type=int, default=100)

    args = parser.parse_args([
        '--data-source', 'hf_tar_stream',
        '--hf-repo', 'username/dataset',
        '--hf-file-path', 'train.tar.gz',
        '--buffer-size', '2000'
    ])

    config = StreamingConfig.from_args(args)
    config.validate()

    print(f"Config: {config}")
    print(f"Cache key: {config.get_cache_key()}")


# Example 8: Local caching to avoid re-streaming
def example_caching():
    """Stream data and cache locally for faster subsequent runs."""
    import tempfile
    from data.tar_streaming import StreamingConfig, StreamingPAPDataset

    # First run: stream from HF Hub and cache locally
    with tempfile.TemporaryDirectory() as tmpdir:
        config = StreamingConfig(
            data_source='hf_tar_stream',
            repo_id='username/dataset',
            file_path='train.tar.gz',
            buffer_size=1000,
            batch_size=32,
            cache_dir=tmpdir,   # Cache records here
            log_interval=50,   # Log every 50 records
            max_retries=5,     # Retry up to 5 times on network errors
        )

        dataset = StreamingPAPDataset(config=config, shuffle=True)
        loader = DataLoader(dataset, batch_size=32)

        # First iteration: streams from HF Hub, writes cache
        for batch_idx, (images, labels) in enumerate(loader):
            if batch_idx >= 2:
                break
        print(f"After streaming: {dataset.stats()}")

        # Second iteration: reads from cache (much faster, no network)
        dataset2 = StreamingPAPDataset(config=config, shuffle=True)
        loader2 = DataLoader(dataset2, batch_size=32)
        for batch_idx, (images, labels) in enumerate(loader2):
            if batch_idx >= 2:
                break
        print(f"After cache read: {dataset2.stats()}")

    # Cache is automatically cleaned up when tmpdir is deleted
    # In production, cache persists in the specified cache_dir


# Example 9: Error handling and retry configuration
def example_error_handling():
    """Configure retry behavior for unreliable networks."""
    from data.tar_streaming import StreamingConfig, StreamingPAPDataset

    config = StreamingConfig(
        data_source='hf_tar_stream',
        repo_id='username/dataset',
        file_path='train.tar.gz',
        max_retries=5,       # Retry up to 5 times (exponential backoff)
        log_interval=10,     # Log frequently for debugging
        buffer_size=500,     # Smaller buffer for low-memory environments
    )

    # The streamer will:
    # 1. Retry up to max_retries times with exponential backoff (2s, 4s, 8s, ...)
    # 2. Skip corrupted tar members with a warning log
    # 3. Log progress every log_interval records
    # 4. Track processed/error/skipped counters accessible via dataset.stats()

    dataset = StreamingPAPDataset(config=config, shuffle=True)
    print(f"Retry config: max_retries={config.max_retries}")
    print(f"Cache config: cache_dir={config.cache_dir}")

    # Check streamer stats after iteration
    # stats = dataset.stats()
    # print(f"Processed: {stats['streamer']['processed']}")
    # print(f"Errors: {stats['streamer']['errors']}")
    # print(f"Skipped: {stats['streamer']['skipped']}")


if __name__ == '__main__':
    print("HF Tar Streaming Examples")
    print("=" * 60)

    print("\n1. Basic Streaming")
    example_basic_streaming()

    print("\n2. YAML Config")
    loader = example_yaml_config()

    print("\n3. Convenience Wrapper")
    loader = example_convenience_wrapper()

    print("\n4. Direct Streamer Usage")
    example_direct_streamer()

    print("\n5. Training Loop")
    example_training_loop()

    print("\n6. Validation")
    example_validation()

    print("\n7. CLI Integration")
    example_cli_integration()

    print("\n8. Caching")
    example_caching()

    print("\n9. Error Handling")
    example_error_handling()

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("\nTo use streaming in your own code:")
    print("  from data.tar_streaming import StreamingConfig, StreamingPAPDataset")
    print("\nFor training with streaming:")
    print("  python vispr/tools/scripts/train_torch.py \\")
    print("      --data-source hf_tar_stream \\")
    print("      --hf-repo username/dataset \\")
    print("      --hf-file-path train.tar.gz \\")
    print("      --cache-dir ./stream_cache \\")
    print("      --max-retries 5")
