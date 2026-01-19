# DINOv3 重新训练指南

## 问题背景

发现了一个严重的bug：`in_chans` 参数没有从config传递到模型构造函数，导致：
- 即使config设置了 `in_chans: 1`，模型仍然使用默认的 `in_chans: 3`
- 如果数据预处理的normalization设置与实际通道数不匹配，会严重影响训练效果

**Bug已修复**，位于 `dinov3/models/__init__.py`

---

## 重新训练前检查清单

### 1. 配置文件检查

确保你的训练config（`.yaml`文件）中以下设置一致：

```yaml
student:
  in_chans: 1  # 或 3，取决于你的数据

teacher:
  in_chans: 1  # 可选，如果不设置会使用student的值

crops:
  rgb_mean: [0.5]      # 单通道：一个值；三通道：三个值
  rgb_std: [0.25]      # 必须和rgb_mean长度一致
```

**关键原则：**
- `in_chans` 必须匹配你的实际数据通道数
- `rgb_mean` 和 `rgb_std` 的长度必须等于 `in_chans`
- 如果数据是灰度图 → `in_chans: 1`, `rgb_mean: [value]`
- 如果数据是RGB → `in_chans: 3`, `rgb_mean: [r, g, b]`

### 2. 运行配置验证脚本

在开始训练前，**必须**运行验证脚本：

```bash
python verify_training_config.py --config path/to/your/config.yaml
```

这个脚本会检查：
- ✓ in_chans 配置一致性
- ✓ normalization 统计量与通道数匹配
- ✓ 模型架构参数合理性
- ✓ 模型实例化成功且patch_embed.in_chans正确
- ✓ 训练超参数在合理范围内

**可选**：运行forward pass测试（需要更多内存）：
```bash
python verify_training_config.py --config path/to/your/config.yaml --test-forward
```

**只有当所有检查都通过（显示绿色✓），才能开始训练！**

---

## 训练健康信号检查

### 启动训练后的前100个iteration

训练开始后，监控以下指标判断是否正常：

#### 1. **初始Loss值**（最重要）

- **正常范围**: 10-12 左右（假设使用65536个prototypes）
- **计算公式**: 初始loss ≈ log(num_prototypes)
  - 65536 prototypes → log(65536) ≈ 11.09
  - 如果你的config设置不同，相应调整期望值

**信号：**
- ✓ **健康**: Loss在9-13之间
- ⚠️ **异常**: Loss < 5 或 > 15
- ✗ **严重**: Loss是NaN或Inf

#### 2. **Loss下降趋势**

前100个iteration应该看到明显下降：

```
Iteration 1:   loss = 11.2
Iteration 10:  loss = 10.8
Iteration 50:  loss = 9.5
Iteration 100: loss = 8.2
```

**信号：**
- ✓ **健康**: 在100 iterations内loss下降 > 2.0
- ⚠️ **缓慢**: loss下降 < 1.0（可能配置问题）
- ✗ **停滞**: loss几乎不变或上升

#### 3. **学习率Warmup**

检查学习率是否按照warmup schedule增加：

```yaml
optim:
  lr: 0.001           # 最终学习率
  warmup_epochs: 10   # warmup周期
```

**信号：**
- ✓ **健康**: 前几个epoch LR从小值逐渐增加到设定值
- ✗ **异常**: LR一开始就是最大值（warmup未生效）

#### 4. **GPU内存使用**

- **稳定性**: 前100 iterations内GPU内存应该稳定
- **首次allocation**: 第一个iteration会分配大量内存（正常）
- **后续iterations**: 内存使用应该保持恒定（±0.5GB）

**信号：**
- ✓ **健康**: 内存稳定在某个值
- ⚠️ **警告**: 内存持续缓慢增长（可能内存泄漏）
- ✗ **严重**: 内存快速增长导致OOM

#### 5. **梯度健康**

如果config开启了 `train.monitor_gradient_norm: true`：

```yaml
train:
  monitor_gradient_norm: true
```

**信号：**
- ✓ **健康**: 梯度norm在合理范围（0.1-10.0）
- ⚠️ **警告**: 梯度norm > 100（可能需要调整clip_grad）
- ✗ **严重**: 梯度是NaN或0

---

## 实时监控方法

### 方法1：使用monitoring脚本（推荐）

在训练启动后，在另一个终端运行：

```bash
# 实时监控模式
python monitor_training_health.py --log-file path/to/training.log --watch

# 或一次性分析
python monitor_training_health.py --log-file path/to/training.log
```

这个脚本会自动分析并给出健康评估。

### 方法2：手动检查日志

使用 `tail` 命令实时查看训练日志：

```bash
tail -f path/to/training.log | grep -E "(loss|lr|iteration)"
```

### 方法3：TensorBoard（如果启用）

如果你的训练代码记录了TensorBoard：

```bash
tensorboard --logdir path/to/logs
```

监控：
- Loss曲线应该平滑下降
- Learning rate曲线遵循warmup+cosine decay

---

## 快速健康检查表（前100 iterations）

打印这个检查表，在训练开始后逐项确认：

```
□ Iteration 1-10:
  □ 初始loss在9-13范围内
  □ 无NaN或Inf
  □ GPU内存已分配并稳定

□ Iteration 10-50:
  □ Loss开始下降
  □ 学习率在warmup（如果配置了）
  □ 无CUDA OOM错误

□ Iteration 50-100:
  □ Loss持续下降（总降幅 > 2.0）
  □ GPU内存稳定（无持续增长）
  □ 训练速度稳定（iter/s基本恒定）

□ Iteration 100:
  □ 运行验证: python monitor_training_health.py --log-file training.log
  □ 检查是否所有健康指标都通过
```

**如果前100个iterations所有检查都通过 → 训练配置正确，可以放心让它跑下去！**

---

## 异常情况处理

### 问题1: 初始loss过高（> 15）或过低（< 5）

**可能原因：**
- prototypes数量设置错误
- 初始化问题
- normalization配置错误

**解决方案：**
1. 检查 `dino.head_n_prototypes` 设置
2. 重新运行 `verify_training_config.py`
3. 确认数据预处理正确

### 问题2: Loss不下降或上升

**可能原因：**
- 学习率过大或过小
- normalization与数据不匹配
- 数据损坏

**解决方案：**
1. 检查学习率设置（`optim.lr`）
2. 验证 `in_chans` 和 `rgb_mean/std` 一致性
3. 检查数据加载是否正常

### 问题3: 出现NaN

**可能原因：**
- 学习率过大
- 梯度爆炸
- 数值不稳定

**解决方案：**
1. 降低学习率（`optim.lr`）
2. 增加梯度裁剪（`optim.clip_grad`）
3. 检查是否使用了合适的dtype（bf16/fp32）

### 问题4: GPU OOM

**可能原因：**
- batch size过大
- 图像尺寸过大
- 模型过大

**解决方案：**
1. 减小 `train.batch_size_per_gpu`
2. 减小 `crops.global_crops_size`
3. 启用gradient checkpointing（`train.checkpointing: true`）

---

## 完整训练流程

### Step 1: 准备配置
```bash
# 编辑你的config文件，确保in_chans和normalization设置正确
vim path/to/your/config.yaml
```

### Step 2: 验证配置
```bash
# 运行验证脚本
python verify_training_config.py --config path/to/your/config.yaml

# 确保所有检查都通过（绿色✓）
```

### Step 3: 启动训练
```bash
# 启动训练，将输出重定向到日志文件
python dinov3/train/train.py \
    --config path/to/your/config.yaml \
    --output-dir path/to/output \
    2>&1 | tee training.log
```

### Step 4: 实时监控（在另一个终端）
```bash
# 方法1: 使用监控脚本
python monitor_training_health.py --log-file training.log --watch

# 方法2: 手动tail
tail -f training.log
```

### Step 5: 等待100 iterations后检查
```bash
# 运行健康分析
python monitor_training_health.py --log-file training.log

# 如果所有指标健康，可以放心继续训练
```

---

## 常见问题FAQ

**Q1: 我应该什么时候认为训练是"正常"的？**
A: 前100个iterations内loss持续下降，无NaN，无OOM → 配置正确，可以继续

**Q2: 训练多久可以看到有意义的结果？**
A: DINOv3通常需要训练数万到数十万iterations。但前100 iterations就能判断配置是否正确

**Q3: 如果中途想修改配置怎么办？**
A: 停止训练，修改config，重新运行验证脚本，从checkpoint resume

**Q4: 如何从checkpoint恢复训练？**
A: 在config中设置 `student.pretrained_weights: path/to/checkpoint.pth` 或使用 `--resume` 参数

**Q5: 单通道和三通道训练效果差异大吗？**
A: 差异不在通道数本身，而在配置一致性。单通道配置正确可以训练良好；三通道配置错误会完全失败

---

## 工具清单

本目录提供以下工具：

1. **verify_training_config.py** - 训练前配置验证（必须）
2. **monitor_training_health.py** - 训练健康监控（推荐）
3. **convert_distcp_to_pth.py** - checkpoint格式转换
4. **RETRAINING_GUIDE.md** - 本指南（你正在阅读）

---

## 总结

重新训练的关键是**配置一致性**：

```
in_chans = 数据实际通道数 = normalization统计量长度 = 模型patch_embed.in_chans
```

**训练前必做：**
1. ✓ 修改config确保一致性
2. ✓ 运行 `verify_training_config.py`
3. ✓ 所有检查通过才启动训练

**训练后监控：**
1. ✓ 前10个iter检查初始loss
2. ✓ 前100个iter确认loss下降
3. ✓ 运行 `monitor_training_health.py` 确认健康

**如果所有步骤都通过 → 训练配置正确，放心训练！**
