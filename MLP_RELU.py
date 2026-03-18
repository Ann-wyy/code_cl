import os
import sys
sys.path.insert(0, "/data/truenas_B2/yyi/dinov2")
sys.path.insert(0, "/data/dataserver01/zhangruipeng/code/PETCT/dinov3_pretrain/dinov3")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import OmegaConf
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import default_collate
import logging
import time
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as T
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from collections import defaultdict
import nibabel as nib

# ================================导入工具函数====================================
from utils.utils import set_seed, convert_dinov3_teacher_to_hf_state_dict, preprocess_labels_and_setup_datasets
from metrics import calculate_metrics, log_metrics_to_tensorboard, evaluate
from config import (
    DINO_VERSION, DEVICE, TARGET_IMAGE_SIZE, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS,
    PATIENCE, UNFREEZE_LAYERS, RANDOM_SEED, NUM_FOLDS,
    TRAIN_NAME, TRAIN_CSV_PATH, VAL_CSV_PATH, TEST_CSV_PATH,
    IMAGE_PATH_COLUMN, LABEL_COLUMNS, TEXT_COLS,
    LOAD_LOCAL_CHECKPOINT, TEST_NAME, LOCAL_CHECKPOINT_PATH, CFG_PATH,
    IGNORE_INDEX, LOG_DIR, LOG_FILENAME
)
if DINO_VERSION == "v3":
    from dinov3.models import build_model_from_cfg
    from dinov3.checkpointer import init_fsdp_model_from_checkpoint
elif DINO_VERSION == "v2":
    from dinov2.models import build_model_from_cfg
# clinical
# --- 自定义 PyTorch Dataset (处理多列分类标签) ---
class MultiTaskImageDatasetFromDataFrame(Dataset):
    def __init__(self, df: pd.DataFrame, img_col: str, 
                 label_cols: List[str], 
                 size: int, logger: logging.Logger,clinical_encoder, is_training: bool = False):
        self.df = df
        self.img_col = img_col
        self.label_cols = label_cols
        self.size = size
        self.logger = logger
        self.clinical_encoder = clinical_encoder
        cfg = OmegaConf.load(CFG_PATH)
        self.mean = getattr(cfg.crops, "rgb_mean", None) 
        self.std  = getattr(cfg.crops, "rgb_std", None)
        self.processor = T.Compose([
            T.Resize((self.size, self.size)),
            T.ToTensor(),
            T.Normalize(mean=self.mean,std=self.std)
        ])

        if is_training:
            self.transform = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=15),
                T.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=0),
            ])
        else:
            self.transform = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row[self.img_col]
        if str(img_path).endswith(".npy"):
            # 读取 npy
            image = np.load(img_path)  # shape: H x W 或 H x W x C
            image = np.nan_to_num(image, nan=0, posinf=1, neginf=0)
            # 如果是二维灰度图，扩展通道
            if image.max() > 1:
                image = image / 255.0

            if image.ndim == 2:
                image = np.stack([image]*3, axis=-1)

            image = Image.fromarray((image * 255).astype(np.uint8))
        else:
            # 普通图片
            image = Image.open(img_path).convert("RGB")

        
        if self.transform:
            image = self.transform(image)

        pixel_values = self.processor(image)
        if torch.isnan(pixel_values).any():
            print(f"[WARN] {img_path} 归一化后出现 NaN, mean={self.mean}, std={self.std}")

        clinical_values = self.clinical_encoder.encode(row)

        labels_dict = {}
        for task in self.label_cols:
            label_val = row[task]
            # 如果 label_val == -1（未知类别），可在此返回 None 或保留（后续 loss 忽略需特殊处理）
            labels_dict[task] = torch.tensor(label_val, dtype=torch.long)

        return pixel_values, clinical_values, labels_dict, img_path



# ====================================================================
# 2. custom_collate_fn 实现
# ====================================================================



class ClinicalEncoder:
    def __init__(self, df, text_cols):
        # 性别映射
        self.gender_map = {val: i for i, val in enumerate(df['Gender'].unique())}
        '''
        # 部位映射
        self.body_part_map = {val: i for i, val in enumerate(df['BodyPart'].unique())}'''
        # 记录维度
        self.clinical_dim = 1 + len(self.gender_map)

    def encode(self, row):
        # 1. 年龄归一化 (假设最大100岁)
        age_val = row['age']
        if pd.isna(age_val):
            age_val = 50.0 # 或者使用 df['age'].mean()
        age = torch.tensor([float(age_val) / 100.0], dtype=torch.float32)
        # 2. 性别 One-hot
        gender = torch.zeros(len(self.gender_map))
        gender_val = str(row['Gender'])
        if gender_val in self.gender_map:
            gender[self.gender_map[gender_val]] = 1.0
        '''
        # 3. 部位 One-hot
        body = torch.zeros(len(self.body_part_map))
        body_val = str(row['BodyPart'])
        if body_val in self.body_part_map:
            body[self.body_part_map[body_val]] = 1.0'''
        
        return torch.cat([age, gender])


# ---- Gated ---
class GatedFusionHead(nn.Module):
    def __init__(self, image_dim, clinical_dim, output_dim):
        super().__init__()
        # 投影层：将不同模态对齐到同一维度
        self.img_proj = nn.Sequential(nn.Linear(image_dim, 512), nn.ReLU())
        self.cli_proj = nn.Sequential(nn.Linear(clinical_dim, 512), nn.ReLU())
        
        # 门控网络：学习一个权重来平衡两者的重要性
        self.gate = nn.Sequential(
            nn.Linear(512 + 512, 512),
            nn.Sigmoid()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, output_dim)
        )

    def forward(self, img_feat, cli_feat):
        i_p = self.img_proj(img_feat)
        c_p = self.cli_proj(cli_feat)
        # 计算门控值
        g = self.gate(torch.cat([i_p, c_p], dim=1))
        self.last_g = g.detach().clone()
        
        # 融合：如果g趋近1则偏向图像，趋近0则偏向临床
        fused = g * i_p + (1 - g) * c_p
        return self.classifier(fused)

# --- 自定义模型：DINOv3 + 多个分类头 ---
def extract_dino_feature(backbone, pixel_values, dino_version):

    if dino_version == "v3":
        features = backbone.forward_features(pixel_values)
        global_feature = features["x_norm_clstoken"]

    elif dino_version == "v2":
        features = backbone.forward_features(pixel_values)
        global_feature = features['x_norm_clstoken']

    return global_feature

class DinoV3MultiTaskClassifier(nn.Module):
    """
    基于 DINOv3 主干网络，带有多任务分类头。
    只加载本地 checkpoint，不使用 teacher/student 结构。
    """
    def __init__(self, task_num_classes: Dict[str, int], clinical_dim,logger: logging.Logger):
        super().__init__()

        self.task_names = list(task_num_classes.keys())
        cfg = OmegaConf.load(CFG_PATH)
        self.backbone, self.embed_dim = build_model_from_cfg(cfg, only_teacher=True)
        if DINO_VERSION == "v3":
            self.backbone.to_empty(device=DEVICE)
        elif DINO_VERSION == "v2":
            self.backbone.to(device=DEVICE)
        checkpoint = torch.load(LOCAL_CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
        logger.info(f"load checkpoint: {LOCAL_CHECKPOINT_PATH}")
        if DINO_VERSION == "v3":
            state_dict = checkpoint["teacher"] if "teacher" in checkpoint else checkpoint

            model_state_dict = self.backbone.state_dict()
            new_state_dict = {}
            # 获取模型所有的标准键名
            target_keys = model_state_dict.keys()
            for k, v in state_dict.items():
                for tk in target_keys:
                    if k.endswith(tk):
                        new_state_dict[tk] = v
                        break
        elif DINO_VERSION == "v2":
            if "teacher" in checkpoint:
                new_state_dict = checkpoint["teacher"]
            elif "model" in checkpoint:
                new_state_dict = checkpoint["model"]
            else:
                new_state_dict = checkpoint
        msg = self.backbone.load_state_dict(new_state_dict, strict=False)
        logger.info(
            f"Backbone loaded. Missing: {len(msg.missing_keys)}, Unexpected: {len(msg.unexpected_keys)}"
        )
        # ==================== 3. 冻结/解冻主干 ====================
        if UNFREEZE_LAYERS < 0:
            for p in self.backbone.parameters():
                p.requires_grad = True
        else:
            for p in self.backbone.parameters():
                p.requires_grad = False

            num_layers = len(self.backbone.blocks)

            for i in range(num_layers - UNFREEZE_LAYERS, num_layers):
                for p in self.backbone.blocks[i].parameters():
                    p.requires_grad = True

        logger.info(f"已解冻最后 {UNFREEZE_LAYERS} 层 Transformer Blocks")

        # ==================== 4. 定义多任务分类头 ====================
        self.classifiers = nn.ModuleDict()
        feature_dim = self.embed_dim
        for task_name, num_classes in task_num_classes.items():
            out_dim = 1 if num_classes == 2 else num_classes
            self.classifiers[task_name] = GatedFusionHead(feature_dim, clinical_dim, out_dim)

        
    def forward(self, pixel_values: torch.Tensor, clinical_values):
        pixel_values = pixel_values.to(DEVICE)
        clinical_values = clinical_values.to(DEVICE)
        if torch.isnan(pixel_values).any():
            print("pixel_values contain NaN")
        if torch.isinf(pixel_values).any():
            print("pixel_values contain Inf")

        if torch.isnan(clinical_values).any():
            print("clinical_values contain NaN")
        if torch.isinf(clinical_values).any():
            print("clinical_values contain Inf")
        if UNFREEZE_LAYERS == 0:
            with torch.no_grad():
                global_feature = extract_dino_feature(self.backbone,pixel_values,DINO_VERSION)
        else:
            global_feature = extract_dino_feature(self.backbone,pixel_values,DINO_VERSION)
        if torch.isnan(global_feature).any():
            print("global_feature contain NaN")
        if torch.isinf(global_feature).any():
            print("global_feature contain Inf")
        logits = {}
        for task_name in self.task_names:

            task_logits = self.classifiers[task_name](global_feature, clinical_values)

            if torch.isnan(task_logits).any():
                print(f"{task_name} logits contain NaN")

            if torch.isinf(task_logits).any():
                print(f"{task_name} logits contain Inf")

            logits[task_name] = task_logits

        return logits
    
    

