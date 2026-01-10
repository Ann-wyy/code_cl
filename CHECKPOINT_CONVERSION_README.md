# DINOv3 分布式 Checkpoint 转换指南

本指南说明如何将 FSDP 分布式保存的 checkpoint（`.distcp` 文件）转换为完整的 `.pth` 文件。

## 背景

当使用 PyTorch FSDP (Fully Sharded Data Parallel) 训练模型时，checkpoint 会被分片保存为多个文件（你的情况是 6 个 `.distcp` 文件）。这些文件需要合并才能在单卡或其他环境中使用。

## 文件说明

提供了 3 个转换脚本：

| 脚本 | 用途 | 推荐度 |
|------|------|--------|
| `convert_fsdp_checkpoint.py` | 简化版，直接转换 | ⭐⭐⭐ 推荐 |
| `merge_distcp_to_pth.py` | 功能完整，支持多种格式 | ⭐⭐ |
| `load_merged_checkpoint.py` | 加载和验证转换后的文件 | ⭐⭐⭐ 推荐 |

---

## 快速开始

### 方法 1: 使用简化版脚本（推荐）

```bash
# 基本用法
python convert_fsdp_checkpoint.py /path/to/checkpoint_dir

# 指定输出路径
python convert_fsdp_checkpoint.py /path/to/checkpoint_dir --output model.pth

# 转换并验证
python convert_fsdp_checkpoint.py /path/to/checkpoint_dir --verify
```

**示例：**

```bash
# 假设你的 checkpoint 在 checkpoints/dinov3_epoch100/ 目录下
# 包含以下文件:
#   - checkpoint_0.distcp
#   - checkpoint_1.distcp
#   - checkpoint_2.distcp
#   - checkpoint_3.distcp
#   - checkpoint_4.distcp
#   - checkpoint_5.distcp

python convert_fsdp_checkpoint.py checkpoints/dinov3_epoch100/
# 输出: checkpoints/dinov3_epoch100.pth
```

---

### 方法 2: 使用功能完整版脚本

```bash
# 提取 student 模型
python merge_distcp_to_pth.py \
    --checkpoint_dir /path/to/checkpoint_dir \
    --model_key student \
    --output student_model.pth

# 提取 teacher 模型
python merge_distcp_to_pth.py \
    --checkpoint_dir /path/to/checkpoint_dir \
    --model_key teacher \
    --output teacher_model.pth

# 提取 EMA 模型
python merge_distcp_to_pth.py \
    --checkpoint_dir /path/to/checkpoint_dir \
    --model_key model_ema \
    --output ema_model.pth
```

---

## 转换后的使用

### 1. 直接加载 state_dict

```python
import torch

# 加载 checkpoint
checkpoint = torch.load('model.pth', map_location='cpu')

# 查看结构
print(checkpoint.keys())
# 可能输出: dict_keys(['model', 'metadata'])
# 或者直接是 state_dict 的键

# 提取 state_dict
if 'model' in checkpoint:
    state_dict = checkpoint['model']
else:
    state_dict = checkpoint

# 使用 state_dict
model.load_state_dict(state_dict, strict=False)
```

### 2. 加载到 SSLMetaArch 模型

```python
from ssl_meta_arch import SSLMetaArch
import torch

# 创建模型
model = SSLMetaArch(cfg)

# 加载 checkpoint
checkpoint = torch.load('student_model.pth', map_location='cpu')
state_dict = checkpoint.get('model', checkpoint)

# 加载到 student
model.student.load_state_dict(state_dict, strict=False)

# 或者使用提供的脚本
# python load_merged_checkpoint.py load \
#     --checkpoint student_model.pth \
#     --model_key student
```

### 3. 用于评估

```python
from vi_dinov3 import build_official_model_eval
import torch

# 使用 vi_dinov3.py 中的评估函数
model = build_official_model_eval(
    cfg=cfg,
    weights_path='student_model.pth'  # 使用转换后的文件
)

# 模型已经在 eval 模式
model.eval()
```

---

## 高级用法

### 提取特定模型部分

如果转换后的 checkpoint 包含多个模型（student, teacher, gram_teacher），可以单独提取：

```bash
python load_merged_checkpoint.py extract \
    --checkpoint full_checkpoint.pth \
    --output student_only.pth \
    --model_key student
```

### 比较两个 checkpoint

验证转换是否正确：

```bash
python load_merged_checkpoint.py compare \
    --ckpt1 original.pth \
    --ckpt2 converted.pth
```

### 验证转换结果

```python
# 使用 Python 验证
import torch

checkpoint = torch.load('model.pth')

# 检查键
print("Keys:", list(checkpoint.keys()))

# 检查 state_dict 结构
state_dict = checkpoint.get('model', checkpoint)
print(f"Number of parameters: {len(state_dict)}")

# 检查部分权重
for key in list(state_dict.keys())[:5]:
    value = state_dict[key]
    if isinstance(value, torch.Tensor):
        print(f"{key}: shape={value.shape}, dtype={value.dtype}")
```

---

## Checkpoint 结构说明

### FSDP 分布式 checkpoint 结构

```
checkpoint_dir/
├── checkpoint_0.distcp  # rank 0 的分片
├── checkpoint_1.distcp  # rank 1 的分片
├── checkpoint_2.distcp  # rank 2 的分片
├── checkpoint_3.distcp  # rank 3 的分片
├── checkpoint_4.distcp  # rank 4 的分片
├── checkpoint_5.distcp  # rank 5 的分片
└── .metadata            # 可选的元数据文件
```

### 转换后的完整 checkpoint 结构

```python
{
    'model': {
        # 根据 --model_key 参数，可能是以下之一:
        # - student: student 模型的完整 state_dict
        # - teacher: teacher 模型的完整 state_dict
        # - model_ema: EMA 模型的完整 state_dict
        # - gram_teacher: gram teacher 模型的完整 state_dict

        'backbone.patch_embed.proj.weight': Tensor(...),
        'backbone.blocks.0.norm1.weight': Tensor(...),
        'backbone.blocks.0.attn.qkv.weight': Tensor(...),
        # ... 更多参数 ...
    },
    'metadata': {
        'source': '/path/to/checkpoint_dir',
        'model_key': 'student',
        'num_params': 123456789,
    }
}
```

---

## 常见问题

### Q1: 转换失败，提示找不到 `.distcp` 文件

**原因**: checkpoint 目录路径不正确

**解决**:
```bash
# 确保目录包含 .distcp 文件
ls /path/to/checkpoint_dir/*.distcp

# 如果文件在子目录中
python convert_fsdp_checkpoint.py /path/to/checkpoint_dir/actual_dir/
```

### Q2: 加载时提示缺少某些键（missing keys）

**原因**: 这是正常的，因为完整模型可能包含多个部分（student, teacher, heads, loss centers 等）

**解决**:
```python
# 使用 strict=False
model.load_state_dict(state_dict, strict=False)

# 或者只加载特定部分
model.student.load_state_dict(state_dict['student'], strict=False)
```

### Q3: 转换后文件太大

**原因**: checkpoint 可能包含多个模型副本

**解决**: 提取单个模型
```bash
# 只提取 student 模型（通常用于微调或评估）
python merge_distcp_to_pth.py \
    --checkpoint_dir /path/to/checkpoint \
    --model_key student \
    --output student_only.pth
```

### Q4: 如何在代码中使用转换后的 checkpoint？

**答**: 参考 `ssl_meta_arch.py` 的 `init_weights()` 方法：

```python
# 方式 1: 加载到新模型
model = SSLMetaArch(cfg)
checkpoint = torch.load('student_model.pth', map_location='cpu')
model.student.load_state_dict(checkpoint['model'], strict=False)

# 方式 2: 用于 resume 训练
cfg.student.resume_from_teacher_chkpt = 'student_model.pth'
model = SSLMetaArch(cfg)
model.init_weights()  # 会自动加载

# 方式 3: 用于评估（参考 vi_dinov3.py）
model = build_official_model_eval(cfg, weights_path='student_model.pth')
```

### Q5: 不同版本的 PyTorch 兼容性？

**答**:
- PyTorch >= 2.0: 使用 `torch.distributed.checkpoint.load()`
- PyTorch < 2.0: 脚本会自动降级使用 `FileSystemReader`
- 如果都失败，会尝试直接加载 `.distcp` 文件

---

## 性能对比

| Checkpoint 类型 | 存储方式 | 加载速度 | 分布式训练 | 单卡使用 |
|----------------|----------|----------|-----------|----------|
| FSDP `.distcp` | 分片保存 | 快（并行加载） | ✅ | ❌ |
| 完整 `.pth` | 单文件 | 中等 | ⚠️ 需要额外处理 | ✅ |

---

## 注意事项

1. **磁盘空间**: 转换后的完整 checkpoint 会占用更多空间（约为分片总和）
2. **内存使用**: 转换过程需要足够内存加载完整模型（建议 2-3 倍模型大小）
3. **备份**: 转换前建议备份原始 `.distcp` 文件
4. **版本兼容**: 确保 PyTorch 版本 >= 1.13（推荐 >= 2.0）

---

## 完整示例工作流

```bash
# 1. 检查原始 checkpoint
ls -lh checkpoints/dinov3_final/
# 输出:
# checkpoint_0.distcp  120M
# checkpoint_1.distcp  120M
# checkpoint_2.distcp  120M
# checkpoint_3.distcp  120M
# checkpoint_4.distcp  120M
# checkpoint_5.distcp  120M

# 2. 转换为完整 checkpoint
python convert_fsdp_checkpoint.py checkpoints/dinov3_final/ \
    --output dinov3_student.pth \
    --verify

# 3. 验证转换结果
python load_merged_checkpoint.py load \
    --checkpoint dinov3_student.pth \
    --model_key student

# 4. 在代码中使用
python vi_dinov3.py --weights dinov3_student.pth --input test_image.jpg
```

---

## 参考资料

- PyTorch FSDP 文档: https://pytorch.org/docs/stable/fsdp.html
- Distributed Checkpoint: https://pytorch.org/docs/stable/distributed.checkpoint.html
- DINOv3 官方仓库: https://github.com/facebookresearch/dinov3

---

## 技术支持

如果遇到问题：

1. 检查 PyTorch 版本: `python -c "import torch; print(torch.__version__)"`
2. 查看 checkpoint 目录结构: `ls -la /path/to/checkpoint/`
3. 验证文件完整性: 确保所有 `.distcp` 文件都存在
4. 尝试使用 `--verify` 选项查看详细信息

生成时间: 2026-01-10
