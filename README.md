# SuperRev-Fusion: A Reversible Network for Multimodal Medical Image Fusion

This repository contains the implementation of **SuperRev-Fusion**, a novel invertible fusion framework based on MRI SR (ASDI), Learned Lifting Wavelet Decomposition (LWD) and Learned Invertible Blocks (LIB) for multimodal medical image fusion.


## Installation

### Requirements

- Python >= 3.8
- PyTorch >= 1.10.0
- CUDA-enabled GPU (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/SuperRev-Fusion.git
cd SuperRev-Fusion

# Install dependencies
pip install -r requirements.txt

# For CUDA support, install PyTorch from:
# https://pytorch.org/get-started/locally/
```

## Datasets

This work evaluates on three multimodal medical image fusion tasks:

| Task | Modality A | Modality B | Description |
|------|-----------|-----------|-------------|
| MRI-PET | MRI | PET | Brain tumor diagnosis |
| MRI-SPECT | MRI | SPECT | Functional imaging |
| CT-PET | CT | PET | Anatomical-metabolic fusion |

Organize your data as follows:

```
test_data/
├── MRI-PET/
│   ├── MRI/
│   └── PET/
├── MRI-SPECT/
│   ├── MRI/
│   └── SPECT/
└── CT-PET/
    ├── CT/
    └── PET/
```

## Testing

### Quick Start

Edit the configuration parameters in `test_final.py`:

```python
TASK = 'MRI-PET'              # Task: MRI-PET / MRI-SPECT / CT-PET
DATA_ROOT = './test_image'    # Dataset root directory
MODEL_PATH = './model.pt'     # model-MRI.pt / model-CT.pt
OUTPUT_DIR = './results'      # Output directory
DEVICE = 'cuda:0'             # Device: cuda:0 or cpu
```

Run inference:

```bash
python test.py
```

### Pre-trained Models

Pre-trained models for each task are available in the directory ./checkpoints

```bash
# Example
python test.py --task MRI-PET --model_path ./checkpoints/model-MRI.pt
```

## Training

To train the model, run:

```bash
python train.py
```
The training pipeline includes:
- **Multi-objective Loss**: SSIM + MSE (luminance/chrominance) + Gradient + LIB regularization
- **Validation Metrics**: SD + CC + SCD (information preservation), SF + AG (spatial quality)
- **Optimization**: Adam with step learning rate decay

> **Note**: Additional configuration files (config.py) and loss function implementations are required to execute training. Please prepare these components according to your experimental setup.

## Contact

For questions or issues, please open an issue on GitHub or feel free to contact [gxu_bingwu@foxmail.com].
