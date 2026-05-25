"""Complete end-to-end example: Train → Evaluate → Deploy a model.

This script shows the full workflow from data preparation to deployment.
Follow this to understand how to use the VPA PyTorch pipeline.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

# ============================================================================
# STEP 1: Prepare Sample Data
# ============================================================================

def create_sample_data():
    """Create sample annotations and image lists for demonstration."""
    print("\n" + "="*60)
    print("STEP 1: Prepare Sample Data")
    print("="*60)

    output_dir = Path('./sample_data')
    output_dir.mkdir(exist_ok=True)

    # Create sample annotations
    annotations = [
        {'image_path': 'image_1.jpg', 'labels': ['a0_safe'], 'safe': True},
        {'image_path': 'image_2.jpg', 'labels': ['a1_adult'], 'safe': False},
        {'image_path': 'image_3.jpg', 'labels': ['a0_safe', 'a3_violence'], 'safe': False},
    ]

    anno_dir = output_dir / 'annotations'
    anno_dir.mkdir(exist_ok=True)

    anno_paths = []
    for idx, anno in enumerate(annotations):
        anno_file = anno_dir / f'anno_{idx}.json'
        with open(anno_file, 'w') as f:
            json.dump(anno, f)
        anno_paths.append(str(anno_file))

    # Create list files
    with open(output_dir / 'train_list.txt', 'w') as f:
        f.write(anno_paths[0] + '\n')

    with open(output_dir / 'val_list.txt', 'w') as f:
        f.write(anno_paths[1] + '\n')

    with open(output_dir / 'test_list.txt', 'w') as f:
        f.write(anno_paths[2] + '\n')

    print(f"✓ Created sample annotations in {anno_dir}")
    print(f"✓ Created train/val/test lists in {output_dir}")
    return output_dir


# ============================================================================
# STEP 2: Train Model
# ============================================================================

def train_model(data_dir, output_dir='./checkpoints'):
    """Train a model on sample data."""
    print("\n" + "="*60)
    print("STEP 2: Train Model")
    print("="*60)

    Path(output_dir).mkdir(exist_ok=True)

    train_list = data_dir / 'train_list.txt'
    val_list = data_dir / 'val_list.txt'
    model_path = Path(output_dir) / 'model.pth'

    cmd = [
        sys.executable,
        'vispr\\tools\\scripts\\train_torch.py',
        '--infile', str(train_list),
        '--valfile', str(val_list),
        '--arch', 'resnet50',
        '--pretrained',
        '--epochs', '2',  # Short for demo
        '--batch-size', '2',
        '--num-classes', '68',
        '--save-path', str(model_path),
    ]

    print(f"Running: {' '.join(cmd)}\n")
    subprocess.run(cmd, check=False)

    if model_path.exists():
        print(f"✓ Model saved to {model_path}")
        return model_path
    else:
        print("✗ Model training failed")
        return None


# ============================================================================
# STEP 3: Run Inference
# ============================================================================

def run_inference(data_dir, model_path, output_dir='.'):
    """Run inference on test data."""
    print("\n" + "="*60)
    print("STEP 3: Run Inference")
    print("="*60)

    test_list = data_dir / 'test_list.txt'
    output_file = Path(output_dir) / 'predictions.jsonl'

    if not model_path.exists():
        print(f"✗ Model not found: {model_path}")
        return None

    cmd = [
        sys.executable,
        'vispr\\tools\\scripts\\attribute_predict_torch.py',
        '--infile', str(test_list),
        '--outfile', str(output_file),
        '--weights', str(model_path),
        '--arch', 'resnet50',
        '--num-classes', '68',
        '--batch-size', '2',
    ]

    print(f"Running: {' '.join(cmd)}\n")
    subprocess.run(cmd, check=False)

    if output_file.exists():
        print(f"✓ Predictions saved to {output_file}")
        # Show first prediction
        with open(output_file) as f:
            first_pred = json.loads(f.readline())
        print(f"✓ Sample prediction: {json.dumps(first_pred, indent=2)}")
        return output_file
    else:
        print("✗ Inference failed")
        return None


# ============================================================================
# STEP 4: Evaluate
# ============================================================================

def evaluate_predictions(pred_file, output_dir='.'):
    """Evaluate predictions."""
    print("\n" + "="*60)
    print("STEP 4: Evaluate Predictions")
    print("="*60)

    if not pred_file.exists():
        print(f"✗ Predictions file not found: {pred_file}")
        return

    metrics_file = Path(output_dir) / 'metrics.tsv'

    cmd = [
        sys.executable,
        'vispr\\tools\\scripts\\evaluate.py',
        str(pred_file),
        '--class_scores', str(metrics_file),
    ]

    print(f"Running: {' '.join(cmd)}\n")
    subprocess.run(cmd, check=False)

    if metrics_file.exists():
        print(f"✓ Metrics saved to {metrics_file}")


# ============================================================================
# STEP 5: Package Model
# ============================================================================

def package_model(model_path, output_dir='./model_package'):
    """Package model for deployment."""
    print("\n" + "="*60)
    print("STEP 5: Package Model for Deployment")
    print("="*60)

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Copy model
    import shutil
    dst = output_dir / 'model.pth'
    shutil.copy(model_path, dst)
    print(f"✓ Model copied to {dst}")

    # Create metadata
    metadata = {
        'model_name': 'VPA Attribute Predictor',
        'architecture': 'resnet50',
        'num_classes': 68,
        'input_shape': [1, 3, 224, 224],
        'input_mean': [104, 117, 123],
        'training_date': '2026-05-17',
    }

    metadata_file = output_dir / 'metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata saved to {metadata_file}")

    return output_dir


# ============================================================================
# STEP 6: Local Inference
# ============================================================================

def demo_local_inference(model_path):
    """Demonstrate local inference using Python API."""
    print("\n" + "="*60)
    print("STEP 6: Local Inference Demo")
    print("="*60)

    try:
        from examples.inference_examples import AttributePredictor

        print("Loading predictor...")
        predictor = AttributePredictor(str(model_path), num_classes=68)
        print("✓ Predictor loaded successfully")

        # Show how to use it
        print("\nExample usage:")
        print("""
# Single image
probs = predictor.predict_image('image.jpg')
print(probs)  # Array of shape (68,)

# From annotation file
result = predictor.predict_from_annotation('anno.json')
print(result['pred_probs'])

# Batch processing
for img_path, probs in predictor.predict_images(image_paths, batch_size=32):
    print(f"{img_path}: top class = {np.argmax(probs)}")
        """)

    except Exception as e:
        print(f"✗ Could not demo local inference: {e}")


# ============================================================================
# Main Workflow
# ============================================================================

def main():
    """Run complete end-to-end workflow."""
    print("\n")
    print("*" * 60)
    print("* VPA PyTorch Complete Workflow Demo")
    print("*" * 60)

    # Step 1: Prepare data
    data_dir = create_sample_data()

    # Step 2: Train
    model_path = train_model(data_dir)
    if not model_path:
        print("\n✗ Training failed, stopping")
        return

    # Step 3: Inference
    pred_file = run_inference(data_dir, model_path)
    if not pred_file:
        print("\n✗ Inference failed, stopping")
        return

    # Step 4: Evaluate
    evaluate_predictions(pred_file)

    # Step 5: Package
    package_dir = package_model(model_path)

    # Step 6: Demo local inference
    demo_local_inference(model_path)

    # Summary
    print("\n" + "="*60)
    print("WORKFLOW COMPLETE")
    print("="*60)
    print(f"""
✓ Data prepared in: {data_dir}
✓ Model trained: {model_path}
✓ Predictions generated
✓ Metrics computed  
✓ Model packaged in: {package_dir}

Next Steps:
1. Review TRAINING_AND_DEPLOYMENT_GUIDE.md for detailed instructions
2. Check examples/inference_examples.py for more usage patterns
3. Train on your own data:
   
   python vispr\\tools\\scripts\\train_torch.py \\
       --infile your_train_list.txt \\
       --valfile your_val_list.txt \\
       --epochs 20 \\
       --save-path ./my_model.pth
    """)


if __name__ == '__main__':
    main()

