# DINOv3 Vision Transformer with PCA Visualization

This repository contains code for evaluating DINOv3 models and visualizing features using PCA.

## Structure

```
.
├── vi_dinov3.py              # Main evaluation and PCA visualization script
├── ssl_meta_arch.py          # Modified SSLMetaArch with gram loss
├── dinov3/                   # Official DINOv3 module from Facebook Research
│   ├── configs/              # Configuration utilities
│   ├── models/               # Model architectures
│   ├── train/                # Training utilities
│   ├── distributed/          # Distributed training support
│   └── ...
├── dinov3_requirements.txt   # DINOv3 dependencies
└── DINOV3_LICENSE.md         # DINOv3 license (Meta Platforms, Inc.)

```

## Features

- **DINOv3 Model Evaluation**: Load and evaluate DINOv3 models in inference mode
- **PCA Visualization**: Extract patch tokens and visualize features using 3-component PCA
- **Distributed Support**: Single GPU/CPU inference with automatic distributed initialization
- **X-ray Image Support**: Specialized preprocessing for medical X-ray images

## Usage

### Prerequisites

```bash
pip install -r dinov3_requirements.txt
```

Required packages:
- torch
- numpy
- pillow
- scikit-learn
- tqdm
- omegaconf

### Running the Script

1. Update the paths in `vi_dinov3.py` (line 129-132):
   - `CONFIG_PATH`: Path to your DINOv3 config YAML file
   - `WEIGHTS_PATH`: Path to your trained model checkpoint
   - `IMAGE_LIST`: List of image paths to process
   - `save_dir`: Output directory for PCA visualizations

2. Run the script:
```bash
python vi_dinov3.py
```

### Key Functions

#### `build_official_model_eval(config_path, weights_path)`
- Initializes distributed environment for single GPU/CPU inference
- Loads DINOv3 model configuration
- Loads trained weights from checkpoint
- Returns model in evaluation mode

#### `run_dinov3_official_pca(config_path, weights_path, image_paths, save_dir, image_size, local_name)`
- Runs PCA visualization pipeline on specified images
- Extracts patch tokens from teacher backbone
- Applies PCA projection to 3 components
- Saves RGB visualization of feature space

## Configuration

### Model Configuration
The model expects a YAML configuration file with DINOv3 architecture settings. Key parameters:
- `student.arch`: Architecture type (e.g., ViT-B/16)
- `crops`: Crop settings for data augmentation
- `dino`: DINO loss configuration
- `ibot`: iBOT loss configuration

### Image Preprocessing
Default preprocessing for X-ray images:
- Mean: `[0.351, 0.35, 0.351]`
- Std: `[0.297, 0.298, 0.298]`

Adjust these values in the `preprocess()` function (line 88-89) for different image types.

### Token Indexing
The script assumes register tokens are used (line 108):
```python
patch_tokens = outputs[:, 5:]  # Skip CLS + 4 register tokens
```

If your model doesn't use register tokens, change to:
```python
patch_tokens = outputs[:, 1:]  # Skip only CLS token
```

## Credits

- **DINOv3**: This repository includes the official DINOv3 implementation from Meta Platforms, Inc.
  - Original repository: https://github.com/facebookresearch/dinov3
  - License: See `DINOV3_LICENSE.md`

## License

- DINOv3 components: Licensed under Meta Platforms, Inc. license (see `DINOV3_LICENSE.md`)
- Custom code (`vi_dinov3.py`, `ssl_meta_arch.py`): Project-specific implementation
