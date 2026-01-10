#!/usr/bin/env python3
"""
简化版: 将 FSDP 分布式 checkpoint 转换为普通 .pth 格式

这个脚本专门处理 PyTorch FSDP 保存的分布式 checkpoint
"""

import torch
import torch.distributed.checkpoint as dcp
from pathlib import Path
import argparse


def convert_fsdp_checkpoint_simple(checkpoint_path, output_path=None):
    """
    最简单的转换方法 - 使用 PyTorch 2.0+ 的 DCP API

    Args:
        checkpoint_path: 分布式 checkpoint 目录（包含 .distcp 文件）
        output_path: 输出的 .pth 文件路径
    """
    checkpoint_path = Path(checkpoint_path)

    if output_path is None:
        output_path = checkpoint_path.parent / f"{checkpoint_path.name}.pth"

    print(f"转换 FSDP checkpoint: {checkpoint_path}")
    print(f"输出文件: {output_path}")

    # 加载分布式 checkpoint
    state_dict = {}

    try:
        # PyTorch 2.0+ 方法
        dcp.load(
            state_dict=state_dict,
            checkpoint_id=str(checkpoint_path),
            no_dist=True,
        )
        print(f"✓ 使用 torch.distributed.checkpoint.load() 加载成功")

    except Exception as e:
        print(f"尝试使用旧版 API...")
        try:
            # 旧版 API
            from torch.distributed.checkpoint import FileSystemReader
            dcp.load_state_dict(
                state_dict=state_dict,
                storage_reader=FileSystemReader(checkpoint_path),
                no_dist=True,
            )
            print(f"✓ 使用 FileSystemReader 加载成功")
        except Exception as e2:
            print(f"错误: {e2}")
            raise

    # 显示 state_dict 结构
    print("\n加载的 checkpoint 结构:")
    for key in state_dict.keys():
        if isinstance(state_dict[key], dict):
            print(f"  {key}/")
            for subkey in list(state_dict[key].keys())[:5]:
                value = state_dict[key][subkey]
                if isinstance(value, torch.Tensor):
                    print(f"    {subkey}: {value.shape} {value.dtype}")
                else:
                    print(f"    {subkey}: {type(value)}")
            if len(state_dict[key]) > 5:
                print(f"    ... ({len(state_dict[key]) - 5} more keys)")
        elif isinstance(state_dict[key], torch.Tensor):
            print(f"  {key}: {state_dict[key].shape} {state_dict[key].dtype}")
        else:
            print(f"  {key}: {type(state_dict[key])}")

    # 保存为标准 .pth 格式
    print(f"\n保存到 {output_path}...")
    torch.save(state_dict, output_path)

    file_size_mb = output_path.stat().st_size / (1024 ** 2)
    print(f"✓ 转换完成! 文件大小: {file_size_mb:.2f} MB")

    return output_path


def load_and_verify(pth_path):
    """验证转换后的 checkpoint"""
    print(f"\n验证 checkpoint: {pth_path}")

    checkpoint = torch.load(pth_path, map_location='cpu')

    print("Checkpoint 包含以下键:")
    for key in checkpoint.keys():
        if isinstance(checkpoint[key], dict):
            print(f"  {key}: dict with {len(checkpoint[key])} keys")
        elif isinstance(checkpoint[key], torch.Tensor):
            print(f"  {key}: Tensor {checkpoint[key].shape}")
        else:
            print(f"  {key}: {type(checkpoint[key])}")

    # 计算总参数量
    total_params = 0
    for key, value in checkpoint.items():
        if isinstance(value, dict):
            for k, v in value.items():
                if isinstance(v, torch.Tensor):
                    total_params += v.numel()
        elif isinstance(value, torch.Tensor):
            total_params += value.numel()

    print(f"\n总参数量: {total_params:,} ({total_params / 1e6:.2f}M)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="转换 FSDP checkpoint 为标准 .pth 文件")
    parser.add_argument("checkpoint_path", help="FSDP checkpoint 目录路径")
    parser.add_argument("--output", "-o", help="输出文件路径（可选）")
    parser.add_argument("--verify", action="store_true", help="转换后验证文件")

    args = parser.parse_args()

    output = convert_fsdp_checkpoint_simple(args.checkpoint_path, args.output)

    if args.verify:
        load_and_verify(output)
