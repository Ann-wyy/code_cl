#!/usr/bin/env python3
"""
将分布式 checkpoint (.distcp) 合并为完整的 .pth 文件

使用方法:
    python merge_distcp_to_pth.py --checkpoint_dir /path/to/checkpoint_dir --output /path/to/output.pth

或者简化版本（自动检测）:
    python merge_distcp_to_pth.py --checkpoint_dir /path/to/checkpoint_dir
"""

import argparse
import os
import torch
import torch.distributed.checkpoint as dist_cp
from torch.distributed.checkpoint.state_dict_loader import load_state_dict
from pathlib import Path


def merge_distcp_to_pth(checkpoint_dir, output_path=None, model_key='student'):
    """
    将分布式 checkpoint 合并为完整的 .pth 文件

    Args:
        checkpoint_dir: 包含 .distcp 文件的目录路径
        output_path: 输出 .pth 文件的路径（可选，默认在 checkpoint_dir 同级目录）
        model_key: 要提取的模型键名（默认 'student'，也可以是 'teacher', 'model_ema' 等）
    """
    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        raise ValueError(f"Checkpoint 目录不存在: {checkpoint_dir}")

    print(f"正在从 {checkpoint_dir} 加载分布式 checkpoint...")

    # 方法 1: 使用 torch.distributed.checkpoint API (推荐)
    try:
        # 创建一个空的 state_dict 来接收加载的权重
        state_dict = {}

        # 加载分布式 checkpoint
        dist_cp.load_state_dict(
            state_dict=state_dict,
            storage_reader=dist_cp.FileSystemReader(checkpoint_dir),
            no_dist=True,  # 在单进程模式下加载
        )

        print(f"成功加载 checkpoint，包含以下键:")
        for key in state_dict.keys():
            if isinstance(state_dict[key], dict):
                print(f"  - {key} (包含 {len(state_dict[key])} 个子键)")
            else:
                print(f"  - {key}")

    except Exception as e:
        print(f"方法 1 失败: {e}")
        print("尝试方法 2: 直接使用 torch.load...")

        # 方法 2: 如果是旧版本的 FSDP checkpoint，尝试直接加载
        try:
            # 查找所有 .distcp 文件
            distcp_files = sorted(checkpoint_dir.glob("*.distcp"))
            if not distcp_files:
                raise ValueError(f"在 {checkpoint_dir} 中没有找到 .distcp 文件")

            print(f"找到 {len(distcp_files)} 个分片文件:")
            for f in distcp_files:
                print(f"  - {f.name}")

            # 加载并合并所有分片
            merged_state_dict = {}
            for distcp_file in distcp_files:
                print(f"加载 {distcp_file.name}...")
                shard = torch.load(distcp_file, map_location='cpu')

                # 合并 state_dict
                if isinstance(shard, dict):
                    for key, value in shard.items():
                        if key in merged_state_dict:
                            # 如果键已存在，可能需要拼接张量（取决于分片策略）
                            print(f"  警告: 键 '{key}' 已存在，跳过或合并...")
                        else:
                            merged_state_dict[key] = value

            state_dict = merged_state_dict

        except Exception as e2:
            print(f"方法 2 也失败: {e2}")
            raise RuntimeError("无法加载 checkpoint，请检查文件格式")

    # 提取指定的模型权重
    if model_key and model_key in state_dict:
        print(f"\n提取 '{model_key}' 模型权重...")
        model_state_dict = state_dict[model_key]
    else:
        print(f"\n使用完整的 state_dict（未找到 '{model_key}' 键）")
        model_state_dict = state_dict

    # 确定输出路径
    if output_path is None:
        output_path = checkpoint_dir.parent / f"{checkpoint_dir.name}_merged.pth"
    else:
        output_path = Path(output_path)

    # 保存为完整的 .pth 文件
    print(f"\n保存合并后的 checkpoint 到: {output_path}")

    # 创建包含元数据的 checkpoint
    checkpoint = {
        'model': model_state_dict,
        'metadata': {
            'source': str(checkpoint_dir),
            'model_key': model_key,
            'num_params': sum(p.numel() for p in model_state_dict.values() if isinstance(p, torch.Tensor)),
        }
    }

    torch.save(checkpoint, output_path)

    print(f"✓ 成功!")
    print(f"  - 参数总数: {checkpoint['metadata']['num_params']:,}")
    print(f"  - 输出文件: {output_path}")
    print(f"  - 文件大小: {output_path.stat().st_size / 1024**2:.2f} MB")

    return output_path


def load_merged_checkpoint(pth_path, model=None):
    """
    加载合并后的 .pth 文件

    Args:
        pth_path: .pth 文件路径
        model: 可选的模型实例，如果提供则直接加载权重

    Returns:
        如果提供了 model，返回加载后的 model；否则返回 state_dict
    """
    print(f"加载 checkpoint: {pth_path}")
    checkpoint = torch.load(pth_path, map_location='cpu')

    if 'metadata' in checkpoint:
        print(f"Checkpoint 元数据:")
        for key, value in checkpoint['metadata'].items():
            print(f"  - {key}: {value}")

    state_dict = checkpoint.get('model', checkpoint)

    if model is not None:
        print("加载权重到模型...")
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        if missing_keys:
            print(f"缺失的键 ({len(missing_keys)}): {missing_keys[:5]}...")
        if unexpected_keys:
            print(f"未预期的键 ({len(unexpected_keys)}): {unexpected_keys[:5]}...")

        return model
    else:
        return state_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将分布式 checkpoint 合并为完整的 .pth 文件")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="包含 .distcp 文件的目录路径"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出 .pth 文件的路径（可选）"
    )
    parser.add_argument(
        "--model_key",
        type=str,
        default="student",
        help="要提取的模型键名（默认: student）。可选: teacher, model_ema, gram_teacher"
    )

    args = parser.parse_args()

    merge_distcp_to_pth(
        checkpoint_dir=args.checkpoint_dir,
        output_path=args.output,
        model_key=args.model_key
    )
