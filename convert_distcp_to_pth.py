#!/usr/bin/env python3
"""
Convert distributed checkpoint (.distcp) to standard PyTorch checkpoint (.pth)

Usage:
    python convert_distcp_to_pth.py \
        --input <distcp_dir> \
        --output <output.pth> \
        --config <config.yaml>

Example:
    python convert_distcp_to_pth.py \
        --input ./output/eval/1000/sharded_teacher_checkpoint \
        --output ./teacher_checkpoint.pth \
        --config ./output/config.yaml
"""

import argparse
from pathlib import Path
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
import torch.distributed.checkpoint as dcp
import torch.distributed.checkpoint.filesystem as dcpfs
import torch.distributed.checkpoint.state_dict as dcpsd
from omegaconf import OmegaConf
from dinov3.models.vision_transformer import vit_small, vit_base, vit_large, vit_giant2


def load_model_from_config(config_path: str):
    """
    Load model configuration from config file.

    Args:
        config_path: Path to the dinov3 config yaml file

    Returns:
        tuple: (model, model_type, patch_size, img_size)
    """
    cfg = OmegaConf.load(config_path)

    # Extract model parameters from config
    model_arch = cfg.student.arch  # e.g., "vit_base", "vit_large"
    patch_size = cfg.student.patch_size
    img_size = cfg.crops.global_crops_size

    # Apply teacher_to_student_resolution_scale if present
    if "teacher_to_student_resolution_scale" in cfg.crops:
        scale = cfg.crops.teacher_to_student_resolution_scale
        # Teacher uses scaled patch size
        teacher_patch_size = int(patch_size * scale)
    else:
        teacher_patch_size = patch_size

    print(f"Config: arch={model_arch}, patch_size={teacher_patch_size}, img_size={img_size}")

    model_fn_map = {
        "vit_small": vit_small,
        "vit_base": vit_base,
        "vit_large": vit_large,
        "vit_giant2": vit_giant2,
    }

    if model_arch not in model_fn_map:
        raise ValueError(f"Unknown model architecture: {model_arch}. Choose from {list(model_fn_map.keys())}")

    # Create model with config parameters
    model = model_fn_map[model_arch](
        patch_size=teacher_patch_size,
        img_size=img_size,
        init_values=1.0,
        block_chunks=0,
    )

    return model, model_arch, teacher_patch_size, img_size


def convert_distcp_to_pth(
    distcp_dir: str,
    output_path: str,
    config_path: str,
):
    """
    Convert a distributed checkpoint to a standard PyTorch checkpoint.

    Args:
        distcp_dir: Path to the distributed checkpoint directory
        output_path: Path to save the output .pth file
        config_path: Path to dinov3 config yaml file
    """
    distcp_dir = Path(distcp_dir)
    output_path = Path(output_path)

    if not distcp_dir.exists():
        raise FileNotFoundError(f"Distributed checkpoint not found: {distcp_dir}")

    print(f"Converting distributed checkpoint from: {distcp_dir}")
    print(f"Output will be saved to: {output_path}")

    # Initialize distributed process (required for loading DCP)
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", init_method="tcp://localhost:12355", rank=0, world_size=1)

    # Load model from config
    print(f"Loading model configuration from: {config_path}")
    model, model_type, patch_size, img_size = load_model_from_config(config_path)

    # Load the distributed checkpoint
    print("Loading distributed checkpoint...")
    to_load = {"model": model.state_dict()}
    dcp.load(
        to_load,
        storage_reader=dcpfs.FileSystemReader(distcp_dir),
        planner=dcp.default_planner.DefaultLoadPlanner(allow_partial_load=True),
    )

    # Convert DTensor to regular tensor
    print("Converting DTensor to regular tensors...")
    new_state_dict = to_load["model"]
    for k, tensor in list(new_state_dict.items()):
        if isinstance(tensor, DTensor):
            new_state_dict[k] = tensor.full_tensor()

    # Save as standard PyTorch checkpoint
    print(f"Saving checkpoint to: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"teacher": new_state_dict}, output_path)

    print("✓ Conversion completed successfully!")
    print(f"Checkpoint saved to: {output_path}")

    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(
        description="Convert distributed checkpoint to standard PyTorch checkpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python convert_distcp_to_pth.py \\
      --input ./output/eval/1000/sharded_teacher_checkpoint \\
      --output ./teacher_checkpoint.pth \\
      --config ./output/config.yaml
        """
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Path to distributed checkpoint directory")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for .pth file")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to dinov3 config yaml file")

    args = parser.parse_args()

    convert_distcp_to_pth(
        distcp_dir=args.input,
        output_path=args.output,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
