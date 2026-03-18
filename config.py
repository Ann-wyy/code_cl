# config.py
import os
import time

# ----------------- 模型与训练参数 -----------------
DINO_VERSION = "v3"            # "v2" 或 "v3"
DEVICE = "cuda:6"              # GPU设备
TARGET_IMAGE_SIZE =  512       # 图像尺寸
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
PATIENCE = 10                  # 早停耐心值
UNFREEZE_LAYERS = 12           # -1 全部解冻, 0 全部冻结, 正数表示解冻最后N层

# ----------------- 数据路径 -----------------
RANDOM_SEED = 42
NUM_FOLDS = 5
TRAIN_NAME = "CancerBenign"
CSV_DIR = "/data/truenas_B2/yyi/datapath/6y_Test_Mask_2_2/Test"
TRAIN_CSV_PATH = os.path.join(CSV_DIR, TRAIN_NAME, f"{TRAIN_NAME}_{RANDOM_SEED}_train.csv")
VAL_CSV_PATH   = os.path.join(CSV_DIR, TRAIN_NAME, f"{TRAIN_NAME}_{RANDOM_SEED}_val.csv")
TEST_CSV_PATH  = os.path.join(CSV_DIR, TRAIN_NAME, f"{TRAIN_NAME}_{RANDOM_SEED}_test.csv")
IMAGE_PATH_COLUMN = 'npy_path'
LABEL_COLUMNS = ["Benign","Intermediate","Malignant"]
TEXT_COLS = ['age','Gender']

# ----------------- checkpoint与日志 -----------------
LOAD_LOCAL_CHECKPOINT = True
if LOAD_LOCAL_CHECKPOINT:
    TEST_NAME = f"xrayDinov3_MLP_test_{DINO_VERSION}_{UNFREEZE_LAYERS}"
else:
    TEST_NAME = "Dinov3"
TEST_NAME = f"{TEST_NAME}_{TRAIN_NAME}_{TARGET_IMAGE_SIZE}_{LEARNING_RATE}_{RANDOM_SEED}"
LOCAL_CHECKPOINT_PATH = "/data/truenas_B2/yyi/weight/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
CFG_PATH = "/home/yyi/CODE/model/dinov3_vitb16_pretrain.yaml"
IGNORE_INDEX = -1

LOG_DIR = f"/data/truenas_B2/yyi/logs/{TRAIN_NAME}/{TEST_NAME}"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILENAME = os.path.join(LOG_DIR, f"{TEST_NAME}_{time.strftime('%Y%m%d-%H%M%S')}.log")