import os
import sys
import torch
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from tqdm import tqdm

# --- 导入官方组件 ---
from dinov3.configs import setup_config
from dinov3.train.ssl_meta_arch import SSLMetaArch
import dinov3.distributed as distributed

# 模拟 argparse 给 setup_config 用
def build_official_model_eval(config_path, weights_path):
    # 设置使用的 GPU 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(0)  # 使用第一个 GPU

    # 初始化分布式环境（即使是单GPU也需要）
    if not distributed.is_enabled():
        # 设置单GPU环境变量
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['LOCAL_RANK'] = '0'

        # 使用 dinov3 的分布式包装器初始化（它会自动调用 init_process_group）
        distributed.enable(overwrite=True)
        print("Distributed environment initialized for single GPU/CPU inference")

    class MockArgs:
        def __init__(self):
            self.config_file = config_path
            # 这里的 opts 配合 eval 模式
            self.opts = []
            self.output_dir = os.path.dirname(config_path)
            self.no_resume = True
            self.eval_only = True  # 触发 eval 模式标志

    args = MockArgs()
    cfg = setup_config(args, strict_cfg=False)

    print("Building model in EVAL mode...")
    # 实例化模型（DINOv3 可能使用 meta 设备初始化）
    model = SSLMetaArch(cfg)

    # 官方推荐：在 eval 模式下手动触发 init_weights
    # 这会初始化 teacher 和 student 的结构
    model.init_weights()

    print(f"Loading checkpoint from: {weights_path}")
    checkpoint = torch.load(weights_path, map_location="cpu")

    # 调试：打印 checkpoint 的顶层键
    print(f"Checkpoint top-level keys (first 10): {list(checkpoint.keys())[:10]}")

    # 检查 checkpoint 的结构并加载权重
    if 'teacher' in checkpoint:
        # 完整的训练checkpoint，包含 teacher/student/model_ema
        state_dict = checkpoint
    elif any(key.startswith('backbone.') for key in list(checkpoint.keys())[:20]):
        # 只有单个模型的权重（键名像 'backbone.xxx'），需要添加前缀映射到 model_ema
        print("Detected single model checkpoint, mapping to model_ema...")
        state_dict = {}
        for key, value in checkpoint.items():
            # 将 backbone.* -> model_ema.backbone.*
            # 将 dino_head.* -> model_ema.dino_head.*
            new_key = f"model_ema.{key}"
            state_dict[new_key] = value
    else:
        # 未知格式
        print(f"WARNING: Unknown checkpoint format! Keys: {list(checkpoint.keys())[:5]}")
        state_dict = checkpoint

    # 加载权重到 CPU
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"Weights loaded. Missing keys: {len(msg.missing_keys)}, Unexpected keys: {len(msg.unexpected_keys)}")

    # 检查是否成功加载了 model_ema 的权重（用于推理）
    if any('model_ema' in key for key in msg.missing_keys):
        print("WARNING: model_ema weights not loaded! Inference may fail.")
    else:
        print("✓ model_ema weights loaded successfully!")

    # 加载权重后再移动到目标设备
    # 这样可以避免 meta tensor 的问题
    print(f"Moving model to {device}...")
    model = model.to(device)
    
    model.eval()
    return model


def run_dinov3_official_pca(
    config_path: str,
    local_weights_path: str,
    image_paths: list[str],
    save_dir: str,
    image_size: int,
    local_name: str
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)

    model = build_official_model_eval(config_path, local_weights_path)

    # 4. 图像预处理 (官方 DINO 通常使用 0.485, 0.456, 0.406 均值)
    def preprocess(img, size):
        img = img.resize((size, size), Image.Resampling.BICUBIC)
        img_arr = np.array(img).astype(np.float32) / 255.0
        # 针对你之前提到的 X-ray 特殊均值（如果有需要可以改回之前的）
        mean = np.array([0.351, 0.35, 0.351])
        std = np.array([0.297, 0.298, 0.298])
        img_arr = (img_arr - mean) / std
        return torch.from_numpy(img_arr).permute(2, 0, 1).unsqueeze(0).to(device)

    # 5. 推理与 PCA
    print("Starting PCA Pipeline...")
    for path in tqdm(image_paths):
        filename = os.path.splitext(os.path.basename(path))[0]
        img_pil = Image.open(path).convert("RGB")
        input_tensor = preprocess(img_pil, image_size)

        with torch.no_grad():
            # 获取 Teacher 网络提取的特征
            # 官方 SSLMetaArch 通常有 backbones['teacher']
            outputs = model.model_ema.backbone(input_tensor)
            
            # DINOv3 结构处理：[Batch, N_patches, Dim]
            # 官方通常 [:, 0] 是 CLS，如果有 registers 则在 1:5
            # 我们取所有 patch tokens
            patch_tokens = outputs[:, 5:] # 根据你是否有 registers 调整
            
            features = patch_tokens.squeeze(0).cpu().numpy()

        # PCA 核心逻辑
        pca = PCA(n_components=3, whiten=True)
        projected = pca.fit_transform(features)
        
        # 重塑回网格尺寸
        side = int(np.sqrt(features.shape[0]))
        projected_img = torch.from_numpy(projected).view(side, side, 3)
        projected_img = torch.sigmoid(projected_img * 2.0).numpy() # 增强对比度

        # 保存结果
        res_img = Image.fromarray((projected_img * 255).astype(np.uint8))
        res_img = res_img.resize(img_pil.size, Image.Resampling.BICUBIC)
        save_path = os.path.join(save_dir, f"{filename}_{local_name}_pca.png")
        res_img.save(save_path)

if __name__ == "__main__":
    # 配置路径
    CONFIG_PATH = "/data/truenas_B2/yyi/train/pretrain_2/dinov3_vitb16_pretrain.yaml" # 换成你真实的 config 路径
    WEIGHTS_PATH = "/data/truenas_B2/yyi/xray_logs_256/eval/training_11249/teacher_checkpoint.pth"
    
    IMAGE_LIST = ["/home/yyi/images/images/spine.png", "/home/yyi/images/images/leg.jpeg"] # ... 其他路径
    
    run_dinov3_official_pca(
        config_path=CONFIG_PATH,
        local_weights_path=WEIGHTS_PATH,
        image_paths=IMAGE_LIST,
        save_dir=f"/home/yyi/images/pca_image/official_v3",
        image_size=1024,
        local_name="xray_v3"
    )
