# 使用自定义数据集训练DINOv3指南

## 已添加的功能

✅ 新增 `NpzDataset` 类，支持：
- CSV文件索引NPZ图像文件
- 单通道/多通道图像
- 灵活的文件路径配置
- 与DINOv3数据增强完全兼容

---

## 数据准备

### 1. NPZ文件格式

每个NPZ文件包含一个图像数组：

```python
import numpy as np

# 单通道图像 (灰度)
image = np.array([[...]])  # shape: (H, W) 或 (H, W, 1)
np.savez('image_001.npz', image=image)

# 三通道图像 (RGB)
image = np.array([[...]])  # shape: (H, W, 3)
np.savez('image_001.npz', image=image)
```

**重要**：
- 默认key是 `'image'`，可以在config中自定义
- 支持uint8 (0-255) 或 float32/float64 (0-1)
- 单通道会自动处理为灰度图

### 2. CSV索引文件

创建CSV文件列出所有NPZ文件路径：

**选项A - 简单列表（推荐）**
```csv
data/train/image_001.npz
data/train/image_002.npz
data/train/image_003.npz
```

**选项B - 带表头**
```csv
path
data/train/image_001.npz
data/train/image_002.npz
data/train/image_003.npz
```

**选项C - 多列（如果有标签，虽然自监督不需要）**
```csv
path,label
data/train/image_001.npz,0
data/train/image_002.npz,1
data/train/image_003.npz,0
```

### 3. 目录结构示例

```
/your/data/root/
├── train.csv              # CSV索引文件
├── val.csv                # 验证集索引（可选）
└── data/
    ├── train/
    │   ├── image_001.npz
    │   ├── image_002.npz
    │   └── ...
    └── val/
        ├── image_001.npz
        └── ...
```

---

## Config配置

### 方法1: 最简单配置（推荐）

在你的训练config YAML中设置：

```yaml
# 单通道数据集
train:
  dataset_path: "NpzDataset:root=/your/data/root:csv_file=train.csv"

student:
  in_chans: 1

teacher:
  in_chans: 1

crops:
  rgb_mean: [0.5]      # 单通道均值
  rgb_std: [0.25]      # 单通道标准差
```

### 方法2: 完整配置（自定义参数）

```yaml
train:
  dataset_path: "NpzDataset:root=/your/data/root:csv_file=train.csv:npz_key=image:image_mode=L"

student:
  in_chans: 1

teacher:
  in_chans: 1

crops:
  rgb_mean: [0.5]
  rgb_std: [0.25]
```

### 方法3: 三通道RGB数据集

```yaml
train:
  dataset_path: "NpzDataset:root=/your/data/root:csv_file=train.csv:image_mode=RGB"

student:
  in_chans: 3

teacher:
  in_chans: 3

crops:
  rgb_mean: [0.485, 0.456, 0.406]  # ImageNet标准
  rgb_std: [0.229, 0.224, 0.225]
```

### 方法4: CSV有表头

```yaml
train:
  dataset_path: "NpzDataset:root=/your/data/root:csv_file=train.csv:has_header=true"

student:
  in_chans: 1

crops:
  rgb_mean: [0.5]
  rgb_std: [0.25]
```

---

## 参数说明

### dataset_path 参数

格式: `NpzDataset:key1=value1:key2=value2:...`

| 参数 | 必需 | 默认值 | 说明 |
|------|------|--------|------|
| `root` | ✅ | - | 数据根目录（包含NPZ文件） |
| `csv_file` | ✅ | - | CSV索引文件路径（相对root或绝对路径） |
| `npz_key` | ❌ | `image` | NPZ文件中图像数组的key |
| `image_mode` | ❌ | `L` | 输出图像模式（`L`=灰度，`RGB`=彩色） |
| `has_header` | ❌ | `false` | CSV是否有表头 |
| `csv_column` | ❌ | `0` | 使用CSV的哪一列作为路径 |

---

## 完整配置示例

### 单通道灰度图训练配置

```yaml
# my_custom_config.yaml

# 基础配置
train:
  dataset_path: "NpzDataset:root=/data/my_dataset:csv_file=train.csv"
  batch_size_per_gpu: 64
  output_dir: /output/my_training

# 学生模型
student:
  arch: vit_base
  patch_size: 16
  in_chans: 1                    # 单通道
  drop_path_rate: 0.1

# 教师模型
teacher:
  in_chans: 1                    # 单通道

# 数据增强
crops:
  global_crops_size: 224         # 或 512, 1024等
  local_crops_size: 96
  local_crops_number: 8
  global_crops_scale: [0.4, 1.0]
  local_crops_scale: [0.05, 0.4]

  # 单通道归一化统计
  rgb_mean: [0.5]                # 根据你的数据调整
  rgb_std: [0.25]                # 根据你的数据调整

# 优化器
optim:
  epochs: 100
  base_lr: 0.0005
  warmup_epochs: 10
  weight_decay: 0.04
```

### 如何计算你的数据集的mean和std

```python
import numpy as np
from pathlib import Path
from tqdm import tqdm

def calculate_dataset_stats(npz_files, npz_key='image', sample_size=1000):
    """计算数据集的均值和标准差"""

    # 随机采样（如果数据集很大）
    if len(npz_files) > sample_size:
        npz_files = np.random.choice(npz_files, sample_size, replace=False)

    all_pixels = []

    for npz_file in tqdm(npz_files):
        data = np.load(npz_file)
        image = data[npz_key]

        # 归一化到 [0, 1]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0

        all_pixels.append(image.flatten())

    all_pixels = np.concatenate(all_pixels)

    mean = np.mean(all_pixels)
    std = np.std(all_pixels)

    print(f"Mean: {mean:.4f}")
    print(f"Std: {std:.4f}")
    print(f"\nConfig设置:")
    print(f"rgb_mean: [{mean:.3f}]")
    print(f"rgb_std: [{std:.3f}]")

    return mean, std

# 使用示例
npz_files = list(Path('/your/data/root/data/train').glob('*.npz'))
calculate_dataset_stats(npz_files)
```

---

## 测试你的数据集

在开始训练前，测试数据集加载是否正常：

```python
# test_dataset.py
import sys
sys.path.insert(0, '/home/user/code_cl')

from dinov3.data.datasets import NpzDataset
from dinov3.data import DataAugmentationDINO

# 创建数据集
dataset = NpzDataset(
    root='/your/data/root',
    csv_file='train.csv',
    npz_key='image',
    image_mode='L',  # 或 'RGB'
)

print(f"Dataset size: {len(dataset)}")
print(f"Loading first sample...")

# 测试加载第一个样本
image, target = dataset[0]
print(f"Image type: {type(image)}")
print(f"Image mode: {image.mode}")
print(f"Image size: {image.size}")
print(f"Target: {target}")

# 测试数据增强
transform = DataAugmentationDINO(
    global_crops_scale=(0.4, 1.0),
    local_crops_scale=(0.05, 0.4),
    local_crops_number=8,
    global_crops_size=224,
    local_crops_size=96,
    mean=[0.5],
    std=[0.25],
)

augmented = transform(image)
print(f"\nAugmented data keys: {augmented.keys()}")
print(f"Global crops: {len(augmented['global_crops'])}")
print(f"Local crops: {len(augmented['local_crops'])}")
print("✓ Dataset test passed!")
```

运行测试：
```bash
python test_dataset.py
```

---

## 完整训练流程

### Step 1: 准备数据

```bash
# 1. 将你的图像转换为NPZ格式
python convert_images_to_npz.py --input /path/to/images --output /data/my_dataset/data/train

# 2. 创建CSV索引
find /data/my_dataset/data/train -name "*.npz" > /data/my_dataset/train.csv

# 3. 计算数据集统计量
python calculate_stats.py --data-root /data/my_dataset --csv train.csv
```

### Step 2: 创建配置文件

复制并修改示例配置：

```bash
cp dinov3/configs/ssl_default_config.yaml my_custom_config.yaml
# 编辑 my_custom_config.yaml，设置 dataset_path 和 in_chans
```

### Step 3: 验证配置

```bash
python verify_training_config.py --config my_custom_config.yaml
```

确保看到：
```
✓ Model patch_embed.in_chans matches config
  → model=1, config=1
```

### Step 4: 测试数据集加载

```bash
python test_dataset.py
```

### Step 5: 启动训练

```bash
python dinov3/train/train.py \
    --config my_custom_config.yaml \
    --output-dir /output/my_training \
    2>&1 | tee training.log
```

### Step 6: 监控训练

```bash
# 在另一个终端
python monitor_training_health.py --log-file training.log --watch
```

---

## 常见问题

### Q1: 我的NPZ文件用了不同的key，怎么办？

A: 在dataset_path中指定：
```yaml
dataset_path: "NpzDataset:root=/data:csv_file=train.csv:npz_key=my_custom_key"
```

### Q2: CSV文件路径是相对还是绝对？

A: 两种都支持：
- 相对路径：相对于config中的 `root` 参数
- 绝对路径：直接使用绝对路径

### Q3: 如何使用绝对路径的CSV？

A: CSV中写绝对路径：
```csv
/absolute/path/to/image_001.npz
/absolute/path/to/image_002.npz
```

Config中root可以随意设置（不会影响绝对路径）

### Q4: 单通道图像但想用RGB模式训练？

A: 设置 `image_mode=RGB`，单通道会自动转换为RGB（三通道相同）
```yaml
dataset_path: "NpzDataset:root=/data:csv_file=train.csv:image_mode=RGB"
in_chans: 3
rgb_mean: [0.5, 0.5, 0.5]
rgb_std: [0.25, 0.25, 0.25]
```

### Q5: NPZ文件很大，加载慢怎么办？

A: 使用内存映射（已默认支持）：
```python
# NpzDataset 内部使用 mmap_mode='r'
data = np.load(npz_path)  # 自动使用内存映射
```

对于超大文件，考虑：
1. 减小图像尺寸
2. 使用更高压缩率
3. 或切换到其他格式（如HDF5）

---

## 高级用法

### 多个数据集混合训练

暂不支持，但可以合并CSV：

```bash
cat dataset1.csv dataset2.csv > combined.csv
```

然后使用 `combined.csv`

### 使用验证集

创建 `val.csv`，在eval时使用：

```python
# 在训练脚本中添加验证数据集
val_dataset = NpzDataset(
    root='/data',
    csv_file='val.csv',
)
```

---

## 总结检查清单

训练前确认：

- [ ] NPZ文件格式正确（包含 'image' key）
- [ ] CSV索引文件包含所有NPZ路径
- [ ] 计算了数据集的mean和std
- [ ] Config中 `in_chans` 与实际通道数匹配
- [ ] Config中 `rgb_mean/std` 长度与 `in_chans` 一致
- [ ] 运行了 `verify_training_config.py` 并通过
- [ ] 运行了 `test_dataset.py` 测试加载
- [ ] 准备好监控脚本

**配置一致性最重要！**

```
数据通道数 = in_chans = len(rgb_mean) = len(rgb_std)
```
