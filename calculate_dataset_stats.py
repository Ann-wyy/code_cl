#!/usr/bin/env python3
"""
Calculate mean and std for custom NPZ dataset.

Usage:
    python calculate_dataset_stats.py --root /data/root --csv train.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm


def load_csv(csv_path, has_header=False):
    """Load NPZ file paths from CSV."""
    paths = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        if has_header:
            next(reader)
        for row in reader:
            if row:
                paths.append(row[0].strip())
    return paths


def calculate_stats(
    root,
    csv_file,
    npz_key="image",
    sample_size=1000,
    has_header=False,
):
    """Calculate dataset mean and standard deviation."""

    print("=" * 80)
    print("  Dataset Statistics Calculation")
    print("=" * 80)

    root = Path(root)
    csv_path = Path(csv_file)
    if not csv_path.is_absolute():
        csv_path = root / csv_path

    # Load file paths
    print(f"\nLoading file paths from: {csv_path}")
    file_paths = load_csv(csv_path, has_header=has_header)
    print(f"Total files: {len(file_paths)}")

    # Convert to absolute paths
    abs_paths = []
    for path_str in file_paths:
        path = Path(path_str)
        if not path.is_absolute():
            path = root / path
        abs_paths.append(path)

    # Sample files if dataset is large
    if len(abs_paths) > sample_size:
        print(f"Sampling {sample_size} files for statistics calculation...")
        indices = np.random.choice(len(abs_paths), sample_size, replace=False)
        sampled_paths = [abs_paths[i] for i in indices]
    else:
        print(f"Using all {len(abs_paths)} files...")
        sampled_paths = abs_paths

    # Collect pixel values
    print("\nLoading images...")
    all_pixels = []
    failed_count = 0

    for npz_path in tqdm(sampled_paths):
        try:
            if not npz_path.exists():
                print(f"\n⚠️  File not found: {npz_path}")
                failed_count += 1
                continue

            data = np.load(npz_path)

            if npz_key not in data:
                print(f"\n⚠️  Key '{npz_key}' not found in {npz_path}")
                print(f"    Available keys: {list(data.keys())}")
                failed_count += 1
                continue

            image = data[npz_key]

            # Normalize to [0, 1] if needed
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
            elif image.dtype in (np.float32, np.float64):
                # Assume already in [0, 1] or similar range
                if image.max() > 1.0:
                    print(f"\n⚠️  Warning: float image with max > 1.0 in {npz_path}")

            # Flatten and collect pixels
            all_pixels.append(image.flatten())

        except Exception as e:
            print(f"\n⚠️  Error loading {npz_path}: {e}")
            failed_count += 1
            continue

    if not all_pixels:
        print("\n✗ No valid images loaded! Check your data.")
        return None, None

    if failed_count > 0:
        print(f"\n⚠️  Failed to load {failed_count} files")

    # Concatenate all pixels
    print("\nCalculating statistics...")
    all_pixels = np.concatenate(all_pixels)

    # Calculate mean and std
    mean = float(np.mean(all_pixels))
    std = float(np.std(all_pixels))

    # Print results
    print("\n" + "=" * 80)
    print("  Results")
    print("=" * 80)
    print(f"\nFiles processed: {len(all_pixels) - failed_count}/{len(sampled_paths)}")
    print(f"Total pixels analyzed: {len(all_pixels):,}")
    print(f"\nMean: {mean:.6f}")
    print(f"Std:  {std:.6f}")

    print("\n" + "-" * 80)
    print("  Config Settings (copy to your YAML)")
    print("-" * 80)
    print(f"\ncrops:")
    print(f"  rgb_mean: [{mean:.4f}]")
    print(f"  rgb_std: [{std:.4f}]")

    # Per-channel statistics if multi-channel
    # Reload one sample to check
    sample_data = np.load(sampled_paths[0])
    sample_image = sample_data[npz_key]

    if sample_image.ndim == 3 and sample_image.shape[2] > 1:
        print("\n" + "-" * 80)
        print("  Multi-channel detected - Per-channel statistics:")
        print("-" * 80)

        channels = sample_image.shape[2]
        channel_means = []
        channel_stds = []

        print("\nRecalculating per-channel statistics...")
        all_images = []

        for npz_path in tqdm(sampled_paths[:100]):  # Use fewer samples for speed
            try:
                data = np.load(npz_path)
                image = data[npz_key]
                if image.dtype == np.uint8:
                    image = image.astype(np.float32) / 255.0
                all_images.append(image)
            except:
                continue

        if all_images:
            all_images = np.stack(all_images)

            for c in range(channels):
                channel_data = all_images[:, :, :, c]
                ch_mean = float(np.mean(channel_data))
                ch_std = float(np.std(channel_data))
                channel_means.append(ch_mean)
                channel_stds.append(ch_std)
                print(f"  Channel {c}: mean={ch_mean:.4f}, std={ch_std:.4f}")

            print(f"\ncrops:")
            mean_str = ", ".join([f"{m:.4f}" for m in channel_means])
            std_str = ", ".join([f"{s:.4f}" for s in channel_stds])
            print(f"  rgb_mean: [{mean_str}]")
            print(f"  rgb_std: [{std_str}]")

    print("\n" + "=" * 80)

    return mean, std


def main():
    parser = argparse.ArgumentParser(
        description="Calculate dataset statistics for normalization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--root", type=str, required=True,
                       help="Data root directory")
    parser.add_argument("--csv", type=str, required=True,
                       help="CSV file path (relative to root or absolute)")
    parser.add_argument("--npz-key", type=str, default="image",
                       help="Key to access image in NPZ file (default: image)")
    parser.add_argument("--sample-size", type=int, default=1000,
                       help="Number of files to sample (default: 1000)")
    parser.add_argument("--has-header", action="store_true",
                       help="CSV file has header row")

    args = parser.parse_args()

    mean, std = calculate_stats(
        root=args.root,
        csv_file=args.csv,
        npz_key=args.npz_key,
        sample_size=args.sample_size,
        has_header=args.has_header,
    )

    if mean is None:
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
