"""Example usage of HF Tar Streaming module.

This file demonstrates various ways to use the streaming data loader.
"""

import torch
from torch.utils.data import DataLoader

# Example 1: Basic streaming usage
def example_basic_streaming():
    """Stream data from HuggingFace Hub - basic usage."""
    from data.tar_streaming import StreamingConfig, StreamingPAPDataset

    # Create configuration
    config = StreamingConfig(
        data_source='hf_tar_stream',
        repo_id='username/vispr-dataset',  # Replace with your repo
        file_path='train.tar.gz',          # Replace with your file
        buffer_size=1000,
        batch_size=32
    )

    # Validate configuration
    config.validate()

    # Create dataset
    dataset = StreamingPAPDataset(config=config, shuffle=True)

    # Create data loader
    loader = DataLoader(dataset, batch_size=32, num_workers=0)

    # Iterate through data
    for batch_idx, (images, labels) in enumerate(loader):
        print(f"Batch {batch_idx}: images shape {images.shape}, labels shape {labels.shape}")
        if batch_idx >= 2:  # Just show first 3 batches
            break


# Example 2: Using YAML configuration
def example_yaml_config():
    """Load configuration from YAML file."""
    from data.tar_streaming import StreamingConfig, StreamingPAPDataset

    # Load config from YAML
    config = StreamingConfig.from_yaml('configs/data_config.yaml')

    # Override specific settings if needed
    config.batch_size = 16
    config.buffer_size = 2000

    # Create dataset
    dataset = StreamingPAPDataset(config=config, shuffle=True)

    # Use with DataLoader
    loader = DataLoader(dataset, batch_size=config.batch_size)

    return loader


# Example 3: Convenience wrapper
def example_convenience_wrapper():
    """Use the convenience wrapper for quick setup."""
    from data.tar_streaming import StreamingPAPDatasetFromFile
    from torch.utils.data import DataLoader

    # Create dataset directly with parameters
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

    # Create streamer
    streamer = HFTarStreamer(
        repo_id='username/dataset',
        file_path='train.tar.gz'
    )

    # Iterate through structured records
    for i, record in enumerate(streamer.extract_structured_data()):
        print(f"Record {i}:")
        print(f"  Image path: {record['image_path']}")
        print(f"  Image size: {record['image'].size}")
        print(f"  Labels: {record.get('labels', [])}")

        if i >= 5:  # Just show first 5 records
            break


# Example 5: Training loop integration
def example_training_loop():
    """Example of integrating with a training loop."""
    import torch.nn as nn
    import torch.optim as optim
    from data.tar_streaming import StreamingConfig, StreamingPAPDataset

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create streaming dataset
    config = StreamingConfig(
        data_source='hf_tar_stream',
        repo_id='username/dataset',
        file_path='train.tar.gz',
        buffer_size=1000,
        batch_size=32
    )

    dataset = StreamingPAPDataset(config=config, shuffle=True)
    loader = DataLoader(dataset, batch_size=32)

    # Create model (simplified example)
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 224 * 224, 68)
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    model.train()
    for epoch in range(2):  # Just 2 epochs for demo
        running_loss = 0.0

        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(device).float()
            labels = labels.to(device).float()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')

            # Stop after a few batches for demo
            if batch_idx >= 20:
                break

        print(f'Epoch {epoch} finished. Avg Loss: {running_loss / (batch_idx + 1):.4f}')


# Example 6: Memory-efficient validation
def example_validation():
    """Example of validation with streaming data."""
    import numpy as np
    from sklearn.metrics import average_precision_score
    from data.tar_streaming import StreamingConfig, StreamingPAPDataset

    # Create validation dataset
    config = StreamingConfig(
        data_source='hf_tar_stream',
        repo_id='username/dataset',
        file_path='val.tar.gz',
        batch_size=32
    )

    dataset = StreamingPAPDataset(config=config, shuffle=False)
    loader = DataLoader(dataset, batch_size=32)

    # Dummy model for example
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

    # Compute metrics
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    # Per-class mAP
    aps = []
    for i in range(all_labels.shape[1]):
        try:
            ap = average_precision_score(all_labels[:, i], all_preds[:, i])
            aps.append(ap)
        except:
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

    # For demo, parse empty args
    args = parser.parse_args([
        '--data-source', 'hf_tar_stream',
        '--hf-repo', 'username/dataset',
        '--hf-file-path', 'train.tar.gz',
        '--buffer-size', '2000'
    ])

    # Create config from args
    config = StreamingConfig.from_args(args)
    config.validate()

    print(f"Config: {config}")


if __name__ == '__main__':
    print("HF Tar Streaming Examples")
    print("=" * 60)

    # Uncomment to run specific examples:

    # print("\n1. Basic Streaming")
    # example_basic_streaming()

    # print("\n2. YAML Config")
    # loader = example_yaml_config()

    # print("\n3. Convenience Wrapper")
    # loader = example_convenience_wrapper()

    # print("\n4. Direct Streamer Usage")
    # example_direct_streamer()

    # print("\n5. Training Loop")
    # example_training_loop()

    # print("\n6. Validation")
    # example_validation()

    print("\n7. CLI Integration")
    example_cli_integration()

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("\nTo use streaming in your own code:")
    print("  from data.tar_streaming import StreamingConfig, StreamingPAPDataset")
    print("\nFor training with streaming:")
    print("  python vispr/tools/scripts/train_torch.py \\")
    print("      --data-source hf_tar_stream \\")
    print("      --hf-repo username/dataset \\")
    print("      --hf-file-path train.tar.gz")
