#!/usr/bin/env python3
"""
Test custom NpzDataset loading.

Usage:
    python test_custom_dataset.py --root /data/root --csv train.csv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dinov3.data.datasets import NpzDataset
from dinov3.data import DataAugmentationDINO


def test_dataset(root, csv_file, npz_key="image", image_mode="L", num_samples=5):
    """Test NpzDataset loading."""

    print("=" * 80)
    print("  Testing NpzDataset")
    print("=" * 80)

    print(f"\nDataset parameters:")
    print(f"  Root: {root}")
    print(f"  CSV file: {csv_file}")
    print(f"  NPZ key: {npz_key}")
    print(f"  Image mode: {image_mode}")

    # Create dataset
    try:
        dataset = NpzDataset(
            root=root,
            csv_file=csv_file,
            npz_key=npz_key,
            image_mode=image_mode,
        )
        print(f"\n✓ Dataset created successfully")
        print(f"  Total samples: {len(dataset)}")
    except Exception as e:
        print(f"\n✗ Failed to create dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test loading samples
    print(f"\nTesting loading {min(num_samples, len(dataset))} samples...")

    for i in range(min(num_samples, len(dataset))):
        try:
            image, target = dataset[i]
            print(f"\n  Sample {i}:")
            print(f"    Image type: {type(image).__name__}")
            print(f"    Image mode: {image.mode}")
            print(f"    Image size: {image.size}")
            print(f"    Target: {target}")

            if i == 0:
                # Test with data augmentation on first sample
                print(f"\n  Testing data augmentation...")

                # Determine mean/std based on image mode
                if image_mode == "L":
                    mean, std = [0.5], [0.25]
                else:
                    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

                transform = DataAugmentationDINO(
                    global_crops_scale=(0.4, 1.0),
                    local_crops_scale=(0.05, 0.4),
                    local_crops_number=8,
                    global_crops_size=224,
                    local_crops_size=96,
                    mean=mean,
                    std=std,
                )

                augmented = transform(image)
                print(f"    Augmented keys: {list(augmented.keys())}")
                print(f"    Global crops: {len(augmented['global_crops'])}")
                print(f"    Local crops: {len(augmented['local_crops'])}")

                # Check tensor shapes
                global_crop = augmented['global_crops'][0]
                local_crop = augmented['local_crops'][0]
                print(f"    Global crop shape: {global_crop.shape}")
                print(f"    Local crop shape: {local_crop.shape}")

                # Verify channel count
                expected_channels = 1 if image_mode == "L" else 3
                actual_channels = global_crop.shape[0]

                if actual_channels != expected_channels:
                    print(f"    ⚠️  Channel mismatch: expected {expected_channels}, got {actual_channels}")
                else:
                    print(f"    ✓ Channels correct: {actual_channels}")

        except Exception as e:
            print(f"\n  ✗ Failed to load sample {i}: {e}")
            import traceback
            traceback.print_exc()
            return False

    print("\n" + "=" * 80)
    print("✓ All tests passed!")
    print("=" * 80)
    print("\nYour dataset is ready for training!")
    print("\nNext steps:")
    print("  1. Calculate dataset mean/std (if not done)")
    print("  2. Update config with correct mean/std values")
    print("  3. Run: python verify_training_config.py --config your_config.yaml")
    print("  4. Start training!")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Test custom NpzDataset loading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--root", type=str, required=True,
                       help="Data root directory")
    parser.add_argument("--csv", type=str, required=True,
                       help="CSV file path (relative to root or absolute)")
    parser.add_argument("--npz-key", type=str, default="image",
                       help="Key to access image in NPZ file (default: image)")
    parser.add_argument("--image-mode", type=str, default="L",
                       choices=["L", "RGB"],
                       help="Image mode: L for grayscale, RGB for color (default: L)")
    parser.add_argument("--num-samples", type=int, default=5,
                       help="Number of samples to test (default: 5)")

    args = parser.parse_args()

    success = test_dataset(
        root=args.root,
        csv_file=args.csv,
        npz_key=args.npz_key,
        image_mode=args.image_mode,
        num_samples=args.num_samples,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
