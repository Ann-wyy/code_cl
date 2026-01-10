#!/usr/bin/env python3
"""
Convert distributed checkpoint (.distcp) to standard PyTorch checkpoint (.pth)

Usage:
    python convert_distcp_to_pth.py --input <distcp_dir> --output <output.pth>

Example:
    python convert_distcp_to_pth.py \
        --input ./output/eval/1000/sharded_teacher_checkpoint \
        --output ./teacher_checkpoint.pth
"""

import argparse
from pathlib import Path
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
import torch.distributed.checkpoint as dcp
import torch.distributed.checkpoint.filesystem as dcpfs
import torch.distributed.checkpoint.state_dict as dcpsd
from dinov3.models.vision_transformer import vit_small, vit_base, vit_large, vit_giant2


def convert_distcp_to_pth(
    distcp_dir: str,
    output_path: str,
    model_type: str = "vit_base",
    patch_size: int = 14,
    img_size: int = 518,
):
    """
    Convert a distributed checkpoint to a standard PyTorch checkpoint.

    Args:
        distcp_dir: Path to the distributed checkpoint directory
        output_path: Path to save the output .pth file
        model_type: Model architecture (vit_small, vit_base, vit_large, vit_giant2)
        patch_size: Patch size used in the model
        img_size: Image size used in the model
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

    # Create model based on type
    model_fn_map = {
        "vit_small": vit_small,
        "vit_base": vit_base,
        "vit_large": vit_large,
        "vit_giant2": vit_giant2,
    }

    if model_type not in model_fn_map:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(model_fn_map.keys())}")

    print(f"Creating model: {model_type} with patch_size={patch_size}, img_size={img_size}")
    model = model_fn_map[model_type](
        patch_size=patch_size,
        img_size=img_size,
        init_values=1.0,
        block_chunks=0,
    )

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
    parser = argparse.ArgumentParser(description="Convert distributed checkpoint to standard PyTorch checkpoint")
    parser.add_argument("--input", type=str, required=True, help="Path to distributed checkpoint directory")
    parser.add_argument("--output", type=str, required=True, help="Output path for .pth file")
    parser.add_argument("--model_type", type=str, default="vit_base",
                        choices=["vit_small", "vit_base", "vit_large", "vit_giant2"],
                        help="Model architecture type")
    parser.add_argument("--patch_size", type=int, default=14, help="Patch size")
    parser.add_argument("--img_size", type=int, default=518, help="Image size")

    args = parser.parse_args()

    convert_distcp_to_pth(
        distcp_dir=args.input,
        output_path=args.output,
        model_type=args.model_type,
        patch_size=args.patch_size,
        img_size=args.img_size,
    )


if __name__ == "__main__":
    main()
