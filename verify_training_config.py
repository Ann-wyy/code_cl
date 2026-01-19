#!/usr/bin/env python3
"""
Pre-training verification script for DINOv3.
Checks configuration consistency before starting training.

Usage:
    python verify_training_config.py --config path/to/your/config.yaml
"""

import argparse
import sys
from pathlib import Path
from omegaconf import OmegaConf
import torch
import torch.nn as nn


def print_section(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_check(passed, message, details=None):
    status = "✓" if passed else "✗"
    color = "\033[92m" if passed else "\033[91m"
    reset = "\033[0m"
    print(f"{color}{status}{reset} {message}")
    if details:
        print(f"  → {details}")
    return passed


def verify_config(config_path):
    """Verify training configuration for consistency."""

    print_section("DINOv3 Training Configuration Verification")
    print(f"Config file: {config_path}\n")

    # Load config
    try:
        cfg = OmegaConf.load(config_path)
        print_check(True, "Config file loaded successfully")
    except Exception as e:
        print_check(False, "Failed to load config file", str(e))
        return False

    all_checks_passed = True

    # ========================================================================
    # 1. Check in_chans consistency
    # ========================================================================
    print_section("1. Input Channels Configuration")

    student_in_chans = cfg.student.in_chans
    print(f"student.in_chans: {student_in_chans}")

    # Check if teacher.in_chans is set (it might not be in config)
    if "in_chans" in cfg.teacher:
        teacher_in_chans = cfg.teacher.in_chans
        print(f"teacher.in_chans: {teacher_in_chans}")

        # Check consistency
        check = print_check(
            student_in_chans == teacher_in_chans,
            "Student and teacher in_chans match",
            f"student={student_in_chans}, teacher={teacher_in_chans}"
        )
        all_checks_passed &= check
    else:
        print_check(True, "teacher.in_chans not set (will use student.in_chans)")

    # ========================================================================
    # 2. Check normalization statistics
    # ========================================================================
    print_section("2. Normalization Statistics")

    rgb_mean = cfg.crops.rgb_mean
    rgb_std = cfg.crops.rgb_std

    print(f"rgb_mean: {rgb_mean}")
    print(f"rgb_std: {rgb_std}")

    # Check length consistency
    mean_len = len(rgb_mean)
    std_len = len(rgb_std)

    check = print_check(
        mean_len == std_len,
        "rgb_mean and rgb_std have same length",
        f"mean_len={mean_len}, std_len={std_len}"
    )
    all_checks_passed &= check

    # Check if normalization stats match in_chans
    check = print_check(
        mean_len == student_in_chans,
        "Normalization length matches in_chans",
        f"norm_len={mean_len}, in_chans={student_in_chans}"
    )
    all_checks_passed &= check

    if mean_len != student_in_chans:
        if mean_len == 1 and student_in_chans == 3:
            print_check(False, "⚠️  WARNING: Single-channel norm for 3-channel model",
                       "PyTorch will broadcast, but this is usually incorrect for RGB")
        elif mean_len == 3 and student_in_chans == 1:
            print_check(False, "⚠️  WARNING: 3-channel norm for single-channel model",
                       "This will cause an error during training")

    # ========================================================================
    # 3. Check model architecture parameters
    # ========================================================================
    print_section("3. Model Architecture")

    arch = cfg.student.arch
    patch_size = cfg.student.patch_size
    img_size = cfg.crops.global_crops_size

    print(f"Architecture: {arch}")
    print(f"Patch size: {patch_size}")
    print(f"Image size: {img_size}")

    # Check if image size is divisible by patch size
    check = print_check(
        img_size % patch_size == 0,
        "Image size divisible by patch size",
        f"{img_size} % {patch_size} = {img_size % patch_size}"
    )
    all_checks_passed &= check

    # ========================================================================
    # 4. Test model instantiation
    # ========================================================================
    print_section("4. Model Instantiation Test")

    try:
        from dinov3.models import build_model

        print("Creating test model with config parameters...")
        model = build_model(cfg.student, only_teacher=True, img_size=img_size, device="meta")
        teacher, embed_dim = model

        print_check(True, "Model created successfully")
        print(f"  → Embed dim: {embed_dim}")

        # Check patch_embed layer
        if hasattr(teacher, 'patch_embed'):
            actual_in_chans = teacher.patch_embed.in_chans
            check = print_check(
                actual_in_chans == student_in_chans,
                "Model patch_embed.in_chans matches config",
                f"model={actual_in_chans}, config={student_in_chans}"
            )
            all_checks_passed &= check

            if actual_in_chans != student_in_chans:
                print_check(False, "⚠️  CRITICAL BUG: in_chans not passed to model!",
                           "This is the bug we just fixed. Make sure you're using the updated code.")
        else:
            print_check(False, "Could not verify patch_embed (unexpected model structure)")

    except Exception as e:
        print_check(False, "Model instantiation failed", str(e))
        all_checks_passed = False

    # ========================================================================
    # 5. Check training parameters
    # ========================================================================
    print_section("5. Training Parameters")

    batch_size = cfg.train.batch_size_per_gpu
    lr = cfg.optim.lr
    weight_decay = cfg.optim.weight_decay
    epochs = cfg.optim.epochs

    print(f"Batch size per GPU: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Weight decay: {weight_decay}")
    print(f"Epochs: {epochs}")

    # Sanity checks
    check = print_check(batch_size > 0, "Batch size is positive")
    all_checks_passed &= check

    check = print_check(lr > 0 and lr < 1.0, "Learning rate in reasonable range",
                       f"lr={lr}")
    all_checks_passed &= check

    # ========================================================================
    # 6. Summary
    # ========================================================================
    print_section("Verification Summary")

    if all_checks_passed:
        print("\n\033[92m✓ ALL CHECKS PASSED!\033[0m")
        print("\nYour configuration appears to be consistent and ready for training.")
        print("\nRecommended monitoring during training:")
        print("  1. Check initial loss values (should be ~log(num_prototypes) ≈ 11 for 65536)")
        print("  2. Monitor that loss decreases steadily in first 100 iterations")
        print("  3. Verify GPU memory usage is stable")
        print("  4. Check that gradients are not NaN")
        print("\nTo start training:")
        print(f"  python dinov3/train/train.py --config {config_path}")
        return True
    else:
        print("\n\033[91m✗ SOME CHECKS FAILED!\033[0m")
        print("\nPlease fix the issues above before starting training.")
        print("Training with these issues will likely fail or produce poor results.")
        return False


def create_dummy_input_test(config_path):
    """Create a dummy input and test forward pass."""
    print_section("Bonus: Dummy Input Forward Pass Test")

    try:
        cfg = OmegaConf.load(config_path)
        from dinov3.models import build_model

        in_chans = cfg.student.in_chans
        img_size = cfg.crops.global_crops_size
        batch_size = 2

        print(f"Creating dummy input: [{batch_size}, {in_chans}, {img_size}, {img_size}]")

        # Create model on CPU for testing
        model = build_model(cfg.student, only_teacher=True, img_size=img_size, device="cpu")
        teacher, embed_dim = model
        teacher.init_weights()  # Initialize weights
        teacher.eval()

        # Create dummy input
        dummy_input = torch.randn(batch_size, in_chans, img_size, img_size)

        print("Running forward pass...")
        with torch.no_grad():
            output = teacher(dummy_input)

        print_check(True, "Forward pass successful!")
        print(f"  → Input shape: {dummy_input.shape}")
        print(f"  → Output shape: {output.shape}")
        print(f"  → Output contains NaN: {torch.isnan(output).any().item()}")

        return True
    except Exception as e:
        print_check(False, "Forward pass test failed", str(e))
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Verify DINOv3 training configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str, required=True,
                       help="Path to training config yaml file")
    parser.add_argument("--test-forward", action="store_true",
                       help="Also run a dummy forward pass test (requires more memory)")

    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"\033[91m✗ Config file not found: {config_path}\033[0m")
        sys.exit(1)

    # Run verification
    passed = verify_config(config_path)

    # Optional forward pass test
    if args.test_forward and passed:
        create_dummy_input_test(config_path)

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
