
# MotionEditor Environment Setup Guide

## System Information

| Item | Value |
|---|---|
| **Cluster** | SCC (BU Shared Computing Cluster) |
| **GPU** | NVIDIA A100 80GB PCIe |
| **NVIDIA Driver** | 595.71.05 |
| **Driver CUDA Version** | 13.2 (backward compatible) |
| **Python** | 3.10.0 |
| **Conda Env** | `me3` |

---

## Core Package Versions

| Package | Version | Notes |
|---|---|---|
| PyTorch | 2.1.1+cu118 | CUDA 11.8 build |
| torchvision | 0.16.1+cu118 | Matches PyTorch 2.1.1 |
| torchaudio | 2.1.1+cu118 | Matches PyTorch 2.1.1 |
| xFormers | 0.0.23+cu118 | Pre-built wheel for CUDA 11.8 |
| diffusers | 0.15.1 | Original repo requirement |
| transformers | 4.25.1 | Original repo requirement |
| accelerate | 0.20.3 | Compatible with diffusers 0.15.1 |
| huggingface-hub | 0.14.1 | Compatible with diffusers 0.15.1 |
| numpy | 1.24.3 | Required for PyTorch 2.1 (not 2.x) |
| setuptools | 80.10.2 | Retains `pkg_resources` (removed in 82+) |

---

## Installation Steps

### 1. Create Conda Environment

```bash
conda create -n me3 python=3.10 -y
conda activate me3
```

### 2. Load CUDA 11.8 Module (SCC)

```bash
module load cuda/11.8
```

### 3. Install PyTorch + xFormers (CUDA 11.8)

```bash
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 \
    --index-url https://download.pytorch.org/whl/cu118

pip install xformers==0.0.23 \
    --index-url https://download.pytorch.org/whl/cu118
```

### 4. Install Core Diffusion Packages

```bash
pip install diffusers==0.15.1 transformers==4.25.1 \
    huggingface-hub==0.14.1 accelerate==0.20.3
```

### 5. Install NumPy (Must be < 2.0)

```bash
pip install numpy==1.24.3
```

### 6. Install setuptools (Must be <= 80.10.2)

```bash
pip install setuptools==80.10.2
```

> ⚠️ **Warning**: setuptools 82+ removed `pkg_resources`, which breaks `accelerate` and other packages.

### 7. Install Other Requirements

```bash
pip install bitsandbytes==0.35.4 einops imageio==2.25.0 omegaconf \
    ftfy opencv-python timm wandb ipdb matplotlib triton \
    progressbar2==4.2.0 Pillow python-slugify addict yapf \
    onnxruntime pycocotools PyYAML requests supervision termcolor \
    nltk fairscale controlnet_aux tensorboard modelcards
```

---

## Running Training

```bash
conda activate me3
module load cuda/11.8
python train_bg.py --config="configs/case-1/train-bg.yaml"
```

---

## Common Issues & Solutions

### 1. `libstdc++ CXXABI_1.3.15` Error

**Cause**: Conda environment conflicts with system GCC libraries.

**Fix**:
```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

---

### 2. `randn_tensor` Import Error

**Cause**: diffusers 0.21.4+ changed the import path.

**Fix**: Use diffusers 0.15.1 (original repo requirement).

```bash
pip install diffusers==0.15.1
```

---

### 3. xFormers CUDA Not Available

**Cause**: PyTorch 2.11.0 + CUDA 13.0 has no pre-built xFormers wheel.

**Fix**: Downgrade to PyTorch 2.1.1 + CUDA 11.8 with matching xFormers.

```bash
pip install torch==2.1.1 --index-url https://download.pytorch.org/whl/cu118
pip install xformers==0.0.23 --index-url https://download.pytorch.org/whl/cu118
```

---

### 4. `pkg_resources` Not Found

**Cause**: setuptools 82+ removed `pkg_resources`.

**Fix**: Pin setuptools to <= 80.10.2.

```bash
pip install setuptools==80.10.2
```

---

### 5. Symlink Issues with Downloaded Models

**Cause**: `huggingface-cli download` creates relative symlinks that break across SCC nodes.

**Fix**: Re-create absolute symlinks or copy actual files.

```bash
# Re-create absolute symlink
ln -sf /projectnb/cs585/students/rexhsu/cache/hub/models--benjamin-paine--stable-diffusion-v1-5/blobs/BLOB_HASH \
    v1-5-pruned-emaonly.ckpt

# Or copy actual file
cp /projectnb/cs585/students/rexhsu/cache/hub/models--benjamin-paine--stable-diffusion-v1-5/blobs/BLOB_HASH \
    v1-5-pruned-emaonly.ckpt
```

---

## Verification Commands

### Check PyTorch + CUDA
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); \
    print(f'CUDA: {torch.version.cuda}'); \
    print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

### Check xFormers
```bash
python -c "import xformers; print(f'xFormers: {xformers.__version__}')"
```

### Check xFormers CUDA Operation
```bash
python -c "
import torch, xformers.ops
q = torch.randn(1, 8, 16, 64).cuda().half()
k = torch.randn(1, 8, 16, 64).cuda().half()
v = torch.randn(1, 8, 16, 64).cuda().half()
out = xformers.ops.memory_efficient_attention(q, k, v)
print('xFormers CUDA: PASSED')
"
```

### Check All Core Packages
```bash
python -c "
import torch, torchvision, torchaudio, xformers
import diffusers, transformers, accelerate, numpy
print(f'PyTorch: {torch.__version__}')
print(f'torchvision: {torchvision.__version__}')
print(f'torchaudio: {torchaudio.__version__}')
print(f'xFormers: {xformers.__version__}')
print(f'diffusers: {diffusers.__version__}')
print(f'transformers: {transformers.__version__}')
print(f'accelerate: {accelerate.__version__}')
print(f'numpy: {numpy.__version__}')
"
```

---

## Model Download

Download Stable Diffusion 1.5 to `./checkpoints`:

```bash
huggingface-cli download benjamin-paine/stable-diffusion-v1-5 \
    --local-dir ./checkpoints/stable-diffusion-v1-5 \
    --local-dir-use-symlinks False
```

Then copy or symlink the checkpoint file:

```bash
cd checkpoints/stable-diffusion-v1-5
cp v1-5-pruned-emaonly.ckpt pytorch_model.bin
```

---

## Notes

- **SCC CUDA Module**: Always run `module load cuda/11.8` before training.
- **Node Consistency**: SCC has multiple compute nodes. Relative symlinks created on one node may break on another. Use absolute paths or copy files.
- **GPU Memory**: A100 80GB is sufficient for MotionEditor training with xFormers enabled.

