"""Example: How to use the trained model for inference in your own code."""
import json
import torch
import torchvision.models as models
import numpy as np
from PIL import Image
from pathlib import Path

from vispr.torch_utils.transformer import SimpleTransformer


class AttributePredictor:
    """Wrapper class for attribute prediction."""

    def __init__(self, weights_path, num_classes=68, device='cpu'):
        """Initialize the predictor with a model checkpoint.

        Args:
            weights_path: Path to .pth file
            num_classes: Number of attributes
            device: 'cpu' or 'cuda'
        """
        self.device = torch.device(device)
        self.num_classes = num_classes

        # Load model
        model = models.resnet50(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

        # Handle checkpoint format
        checkpoint = torch.load(weights_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)

        self.model = model.to(self.device).eval()
        self.transformer = SimpleTransformer(mean=[104, 117, 123])

    def predict_image(self, image_path):
        """Predict attributes for a single image.

        Args:
            image_path: Path to image file

        Returns:
            numpy array of shape (num_classes,) with predictions in [0, 1]
        """
        # Load and preprocess
        img = Image.open(image_path).convert('RGB')
        arr = np.asarray(img)
        preprocessed = self.transformer.preprocess(arr)

        # Forward pass
        with torch.no_grad():
            tensor = torch.from_numpy(preprocessed).unsqueeze(0).float().to(self.device)
            logits = self.model(tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        return probs

    def predict_images(self, image_paths, batch_size=16):
        """Predict attributes for multiple images.

        Args:
            image_paths: List of image file paths
            batch_size: Batch size for processing

        Yields:
            (image_path, probs) tuples
        """
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_arrays = []

            for path in batch_paths:
                img = Image.open(path).convert('RGB')
                arr = np.asarray(img)
                preprocessed = self.transformer.preprocess(arr)
                batch_arrays.append(preprocessed)

            batch_tensor = torch.from_numpy(np.stack(batch_arrays)).float().to(self.device)

            with torch.no_grad():
                logits = self.model(batch_tensor)
                probs_batch = torch.sigmoid(logits).cpu().numpy()

            for path, probs in zip(batch_paths, probs_batch):
                yield path, probs

    def predict_from_annotation(self, anno_path):
        """Predict from an annotation JSON file.

        Args:
            anno_path: Path to annotation JSON

        Returns:
            dict with predictions and annotation info
        """
        with open(anno_path) as f:
            anno = json.load(f)

        image_path = anno['image_path']
        probs = self.predict_image(image_path)

        return {
            'anno_path': str(anno_path),
            'image_path': image_path,
            'pred_probs': probs.tolist(),
            'pred_probs_dict': {f'class_{i}': float(probs[i]) for i in range(len(probs))}
        }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_single_image():
    """Example: Predict on a single image."""
    print("\n=== Example 1: Single Image Inference ===")

    predictor = AttributePredictor(
        weights_path='./checkpoints/model_best.pth',
        num_classes=68,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Predict on single image
    image_path = 'test_image.jpg'
    probs = predictor.predict_image(image_path)

    print(f"Image: {image_path}")
    print(f"Predictions shape: {probs.shape}")
    print(f"Top 5 predictions: {np.argsort(-probs)[:5]}")
    print(f"Top 5 probabilities: {probs[np.argsort(-probs)[:5]]}")


def example_batch_images():
    """Example: Predict on multiple images."""
    print("\n=== Example 2: Batch Image Inference ===")

    predictor = AttributePredictor(
        weights_path='./checkpoints/model_best.pth',
        num_classes=68
    )

    # Get list of images
    image_paths = list(Path('./images').glob('*.jpg'))[:10]  # First 10 images

    print(f"Processing {len(image_paths)} images...")
    for idx, (image_path, probs) in enumerate(predictor.predict_images(image_paths, batch_size=4)):
        if idx % 5 == 0:
            print(f"  [{idx}] {image_path}: top class = {np.argmax(probs)} (prob={probs[np.argmax(probs)]:.3f})")


def example_from_annotation():
    """Example: Predict from annotation JSON."""
    print("\n=== Example 3: Predict from Annotation ===")

    predictor = AttributePredictor(
        weights_path='./checkpoints/model_best.pth',
        num_classes=68
    )

    # Predict from annotation
    result = predictor.predict_from_annotation('./annotation.json')

    print(f"Annotation: {result['anno_path']}")
    print(f"Image: {result['image_path']}")
    print(f"Predictions (first 10): {result['pred_probs'][:10]}")


def example_save_predictions():
    """Example: Save predictions to JSON."""
    print("\n=== Example 4: Save Predictions ===")

    predictor = AttributePredictor(
        weights_path='./checkpoints/model_best.pth',
        num_classes=68
    )

    # Process annotations and save
    anno_paths = list(Path('./annotations').glob('*.json'))
    output_file = 'predictions.jsonl'

    with open(output_file, 'w') as wf:
        for anno_path in anno_paths[:100]:  # First 100
            result = predictor.predict_from_annotation(anno_path)
            wf.write(json.dumps(result) + '\n')

    print(f"✓ Saved predictions to: {output_file}")


def example_with_attribute_names():
    """Example: Predict with human-readable attribute names."""
    print("\n=== Example 5: Predictions with Attribute Names ===")

    predictor = AttributePredictor(
        weights_path='./checkpoints/model_best.pth',
        num_classes=68
    )

    # Load attribute names (you'll need your own mapping)
    attribute_names = {
        0: 'safe',
        1: 'adult_content',
        2: 'violence',
        # ... add more
    }

    probs = predictor.predict_image('test_image.jpg')

    # Get top predictions
    top_indices = np.argsort(-probs)[:5]
    print("Top 5 attributes:")
    for idx in top_indices:
        attr_name = attribute_names.get(idx, f'attr_{idx}')
        print(f"  {attr_name}: {probs[idx]:.4f}")


if __name__ == '__main__':
    # Run examples
    # example_single_image()
    # example_batch_images()
    # example_from_annotation()
    # example_save_predictions()
    # example_with_attribute_names()

    print("Examples prepared! Uncomment the ones you want to run in __main__")

