#!/bin/bash
# 自动转换 FSDP checkpoint 的辅助脚本

set -e

echo "==================================================="
echo "    DINOv3 Checkpoint 自动转换工具"
echo "==================================================="
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 函数：打印彩色消息
print_info() {
    echo -e "${BLUE}[信息]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[成功]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[警告]${NC} $1"
}

print_error() {
    echo -e "${RED}[错误]${NC} $1"
}

# 检查 Python 是否可用
if ! command -v python &> /dev/null; then
    print_error "找不到 Python! 请先安装 Python 3.7+"
    exit 1
fi

print_success "Python 版本: $(python --version)"

# 检查 PyTorch
if ! python -c "import torch" 2>/dev/null; then
    print_error "找不到 PyTorch! 请先安装: pip install torch"
    exit 1
fi

TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
print_success "PyTorch 版本: $TORCH_VERSION"

echo ""
echo "---------------------------------------------------"
echo "步骤 1: 查找 checkpoint 目录"
echo "---------------------------------------------------"

# 如果用户提供了参数，直接使用
if [ -n "$1" ]; then
    CHECKPOINT_DIR="$1"
    print_info "使用提供的路径: $CHECKPOINT_DIR"
else
    # 搜索 .distcp 文件
    print_info "正在搜索 .distcp 文件..."

    DISTCP_FILES=$(find /home/user -name "*.distcp" -type f 2>/dev/null | head -20)

    if [ -z "$DISTCP_FILES" ]; then
        print_error "未找到任何 .distcp 文件"
        echo ""
        echo "请手动指定 checkpoint 目录："
        echo "  bash auto_convert.sh /path/to/checkpoint_dir"
        echo ""
        echo "或者使用 Python 脚本："
        echo "  python convert_fsdp_checkpoint.py /path/to/checkpoint_dir"
        exit 1
    fi

    # 提取目录并去重
    CHECKPOINT_DIRS=$(echo "$DISTCP_FILES" | xargs -n1 dirname | sort -u)

    # 计算找到的目录数量
    NUM_DIRS=$(echo "$CHECKPOINT_DIRS" | wc -l)

    if [ "$NUM_DIRS" -eq 1 ]; then
        # 只找到一个目录，直接使用
        CHECKPOINT_DIR="$CHECKPOINT_DIRS"
        print_success "自动找到 checkpoint 目录: $CHECKPOINT_DIR"
    else
        # 找到多个目录，让用户选择
        print_warning "找到 $NUM_DIRS 个包含 .distcp 的目录:"
        echo ""

        i=1
        declare -a DIR_ARRAY
        while IFS= read -r dir; do
            NUM_FILES=$(find "$dir" -maxdepth 1 -name "*.distcp" | wc -l)
            DIR_SIZE=$(du -sh "$dir" 2>/dev/null | cut -f1)
            echo "  [$i] $dir"
            echo "      文件数: $NUM_FILES, 大小: $DIR_SIZE"
            DIR_ARRAY[$i]="$dir"
            ((i++))
        done <<< "$CHECKPOINT_DIRS"

        echo ""
        read -p "请选择要转换的目录 [1-$NUM_DIRS]: " choice

        if [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -le "$NUM_DIRS" ]; then
            CHECKPOINT_DIR="${DIR_ARRAY[$choice]}"
            print_success "已选择: $CHECKPOINT_DIR"
        else
            print_error "无效的选择"
            exit 1
        fi
    fi
fi

# 验证目录存在
if [ ! -d "$CHECKPOINT_DIR" ]; then
    print_error "目录不存在: $CHECKPOINT_DIR"
    exit 1
fi

# 检查目录中的 .distcp 文件
NUM_DISTCP=$(find "$CHECKPOINT_DIR" -maxdepth 1 -name "*.distcp" | wc -l)
if [ "$NUM_DISTCP" -eq 0 ]; then
    print_error "目录中没有 .distcp 文件: $CHECKPOINT_DIR"
    exit 1
fi

print_success "找到 $NUM_DISTCP 个 .distcp 文件"

echo ""
echo "---------------------------------------------------"
echo "步骤 2: 确定输出路径"
echo "---------------------------------------------------"

# 默认输出路径
DEFAULT_OUTPUT="${CHECKPOINT_DIR}.pth"

echo "默认输出路径: $DEFAULT_OUTPUT"
read -p "是否使用默认路径? [Y/n]: " use_default

if [[ "$use_default" =~ ^[Nn]$ ]]; then
    read -p "请输入输出路径: " OUTPUT_PATH
else
    OUTPUT_PATH="$DEFAULT_OUTPUT"
fi

print_info "输出文件: $OUTPUT_PATH"

# 检查输出文件是否已存在
if [ -f "$OUTPUT_PATH" ]; then
    print_warning "输出文件已存在: $OUTPUT_PATH"
    read -p "是否覆盖? [y/N]: " overwrite
    if [[ ! "$overwrite" =~ ^[Yy]$ ]]; then
        print_info "已取消"
        exit 0
    fi
fi

echo ""
echo "---------------------------------------------------"
echo "步骤 3: 开始转换"
echo "---------------------------------------------------"

# 询问是否验证
read -p "转换后是否验证文件? [Y/n]: " do_verify

VERIFY_FLAG=""
if [[ ! "$do_verify" =~ ^[Nn]$ ]]; then
    VERIFY_FLAG="--verify"
fi

# 执行转换
print_info "正在转换..."
echo ""

python convert_fsdp_checkpoint.py "$CHECKPOINT_DIR" --output "$OUTPUT_PATH" $VERIFY_FLAG

echo ""
echo "==================================================="
print_success "转换完成!"
echo "==================================================="
echo ""

# 显示文件信息
if [ -f "$OUTPUT_PATH" ]; then
    FILE_SIZE=$(du -sh "$OUTPUT_PATH" | cut -f1)
    print_info "输出文件: $OUTPUT_PATH"
    print_info "文件大小: $FILE_SIZE"

    echo ""
    echo "---------------------------------------------------"
    echo "下一步操作"
    echo "---------------------------------------------------"
    echo ""
    echo "1. 在 Python 中加载 checkpoint:"
    echo ""
    echo "   import torch"
    echo "   checkpoint = torch.load('$OUTPUT_PATH', map_location='cpu')"
    echo "   print(checkpoint.keys())"
    echo ""
    echo "2. 加载到 DINOv3 模型:"
    echo ""
    echo "   from ssl_meta_arch import SSLMetaArch"
    echo "   model = SSLMetaArch(cfg)"
    echo "   state_dict = checkpoint.get('model', checkpoint)"
    echo "   model.student.load_state_dict(state_dict, strict=False)"
    echo ""
    echo "3. 用于评估:"
    echo ""
    echo "   python vi_dinov3.py --weights $OUTPUT_PATH"
    echo ""
else
    print_error "输出文件不存在，转换可能失败"
    exit 1
fi
