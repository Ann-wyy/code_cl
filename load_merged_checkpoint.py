#!/usr/bin/env python3
"""
将合并后的 checkpoint 加载到 DINOv3 SSLMetaArch 模型

使用方法:
    python load_merged_checkpoint.py --checkpoint merged.pth --config config.yaml
"""

import argparse
import torch
from pathlib import Path
from ssl_meta_arch import SSLMetaArch


def load_checkpoint_to_model(checkpoint_path, cfg=None, model_key='student', strict=False):
    """
    将合并后的 checkpoint 加载到 SSLMetaArch 模型

    Args:
        checkpoint_path: 合并后的 .pth 文件路径
        cfg: 模型配置（如果为 None，则只加载 state_dict）
        model_key: 要加载的模型部分（'student', 'teacher', 'model_ema', 'gram_teacher'）
        strict: 是否严格匹配所有键

    Returns:
        如果提供 cfg，返回加载了权重的模型；否则返回 state_dict
    """
    checkpoint_path = Path(checkpoint_path)

    print(f"加载 checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # 提取 state_dict
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
        print("检测到结构化 checkpoint (包含 'model' 键)")

        if 'metadata' in checkpoint:
            print("\nCheckpoint 元数据:")
            for key, value in checkpoint['metadata'].items():
                print(f"  {key}: {value}")
    else:
        state_dict = checkpoint
        print("使用原始 state_dict")

    # 分析 state_dict 结构
    print(f"\nState dict 结构:")
    print(f"  顶层键: {list(state_dict.keys())}")

    # 如果 state_dict 包含模型部分的嵌套结构
    if model_key in state_dict:
        print(f"\n提取 '{model_key}' 部分...")
        model_state_dict = state_dict[model_key]
    else:
        print(f"\n警告: 未找到 '{model_key}' 键，使用完整 state_dict")
        model_state_dict = state_dict

    # 显示部分权重信息
    print(f"\n模型权重概览 (前 10 个键):")
    for i, (key, value) in enumerate(list(model_state_dict.items())[:10]):
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape} {value.dtype}")
        else:
            print(f"  {key}: {type(value)}")

    total_keys = len(model_state_dict)
    if total_keys > 10:
        print(f"  ... (还有 {total_keys - 10} 个键)")

    # 如果提供了配置，创建模型并加载
    if cfg is not None:
        print(f"\n创建 SSLMetaArch 模型...")
        model = SSLMetaArch(cfg)

        # 根据 model_key 选择要加载的模型部分
        if model_key == 'student':
            target_model = model.student
        elif model_key == 'teacher' or model_key == 'model_ema':
            target_model = model.model_ema
        elif model_key == 'gram_teacher':
            if hasattr(model, 'gram_teacher'):
                target_model = model.gram_teacher
            else:
                raise ValueError("模型未启用 gram_teacher")
        else:
            # 加载到整个模型
            target_model = model

        print(f"加载权重到 {model_key}...")
        missing_keys, unexpected_keys = target_model.load_state_dict(
            model_state_dict,
            strict=strict
        )

        if missing_keys:
            print(f"\n⚠ 缺失的键 ({len(missing_keys)}):")
            for key in missing_keys[:10]:
                print(f"  - {key}")
            if len(missing_keys) > 10:
                print(f"  ... (还有 {len(missing_keys) - 10} 个)")

        if unexpected_keys:
            print(f"\n⚠ 未预期的键 ({len(unexpected_keys)}):")
            for key in unexpected_keys[:10]:
                print(f"  - {key}")
            if len(unexpected_keys) > 10:
                print(f"  ... (还有 {len(unexpected_keys) - 10} 个)")

        if not missing_keys and not unexpected_keys:
            print("✓ 所有权重完美匹配!")

        return model
    else:
        return model_state_dict


def extract_specific_model(checkpoint_path, output_path, model_key='student'):
    """
    从完整 checkpoint 中提取特定模型部分并单独保存

    Args:
        checkpoint_path: 完整 checkpoint 路径
        output_path: 输出文件路径
        model_key: 要提取的模型键
    """
    print(f"从 {checkpoint_path} 提取 '{model_key}' 模型...")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # 处理嵌套结构
    if 'model' in checkpoint and model_key in checkpoint['model']:
        extracted_state_dict = checkpoint['model'][model_key]
    elif model_key in checkpoint:
        extracted_state_dict = checkpoint[model_key]
    else:
        raise ValueError(f"在 checkpoint 中找不到 '{model_key}'")

    # 保存
    output_checkpoint = {
        'state_dict': extracted_state_dict,
        'model_key': model_key,
    }

    torch.save(output_checkpoint, output_path)
    print(f"✓ 已保存到 {output_path}")

    file_size_mb = Path(output_path).stat().st_size / (1024 ** 2)
    print(f"  文件大小: {file_size_mb:.2f} MB")


def compare_checkpoints(ckpt1_path, ckpt2_path):
    """比较两个 checkpoint 的差异"""
    print(f"比较 checkpoints:")
    print(f"  1: {ckpt1_path}")
    print(f"  2: {ckpt2_path}")

    ckpt1 = torch.load(ckpt1_path, map_location='cpu')
    ckpt2 = torch.load(ckpt2_path, map_location='cpu')

    # 提取 state_dict
    if 'model' in ckpt1:
        ckpt1 = ckpt1['model']
    if 'model' in ckpt2:
        ckpt2 = ckpt2['model']

    keys1 = set(ckpt1.keys())
    keys2 = set(ckpt2.keys())

    print(f"\nCheckpoint 1: {len(keys1)} 个键")
    print(f"Checkpoint 2: {len(keys2)} 个键")

    only_in_1 = keys1 - keys2
    only_in_2 = keys2 - keys1
    common = keys1 & keys2

    if only_in_1:
        print(f"\n仅在 checkpoint 1 中 ({len(only_in_1)}):")
        for key in list(only_in_1)[:5]:
            print(f"  - {key}")

    if only_in_2:
        print(f"\n仅在 checkpoint 2 中 ({len(only_in_2)}):")
        for key in list(only_in_2)[:5]:
            print(f"  - {key}")

    print(f"\n共同的键: {len(common)}")

    # 比较共同键的值
    if common:
        print("\n比较共同键的值...")
        differences = []
        for key in list(common)[:10]:
            v1, v2 = ckpt1[key], ckpt2[key]
            if isinstance(v1, torch.Tensor) and isinstance(v2, torch.Tensor):
                if v1.shape != v2.shape:
                    differences.append(f"{key}: 形状不同 {v1.shape} vs {v2.shape}")
                elif not torch.allclose(v1, v2, rtol=1e-5, atol=1e-8):
                    max_diff = (v1 - v2).abs().max().item()
                    differences.append(f"{key}: 数值不同 (最大差异: {max_diff:.2e})")

        if differences:
            print(f"\n发现差异:")
            for diff in differences:
                print(f"  - {diff}")
        else:
            print("✓ 所有检查的键值都相同!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="加载合并后的 DINOv3 checkpoint")

    subparsers = parser.add_subparsers(dest='command', help='命令')

    # load 命令
    load_parser = subparsers.add_parser('load', help='加载 checkpoint 到模型')
    load_parser.add_argument('--checkpoint', required=True, help='Checkpoint 文件路径')
    load_parser.add_argument('--config', help='模型配置文件（可选）')
    load_parser.add_argument('--model_key', default='student',
                            choices=['student', 'teacher', 'model_ema', 'gram_teacher'],
                            help='要加载的模型部分')

    # extract 命令
    extract_parser = subparsers.add_parser('extract', help='提取特定模型部分')
    extract_parser.add_argument('--checkpoint', required=True, help='输入 checkpoint')
    extract_parser.add_argument('--output', required=True, help='输出文件')
    extract_parser.add_argument('--model_key', default='student', help='要提取的模型键')

    # compare 命令
    compare_parser = subparsers.add_parser('compare', help='比较两个 checkpoint')
    compare_parser.add_argument('--ckpt1', required=True, help='Checkpoint 1')
    compare_parser.add_argument('--ckpt2', required=True, help='Checkpoint 2')

    args = parser.parse_args()

    if args.command == 'load':
        load_checkpoint_to_model(
            checkpoint_path=args.checkpoint,
            cfg=None,  # 需要提供实际配置
            model_key=args.model_key
        )
    elif args.command == 'extract':
        extract_specific_model(
            checkpoint_path=args.checkpoint,
            output_path=args.output,
            model_key=args.model_key
        )
    elif args.command == 'compare':
        compare_checkpoints(args.ckpt1, args.ckpt2)
    else:
        parser.print_help()
