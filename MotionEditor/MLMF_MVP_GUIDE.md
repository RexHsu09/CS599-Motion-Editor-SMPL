# MotionEditor MLMF (Multi-Layer Motion Fusion) Integration - MVP

## 概述

MVP 实现了一个简单的 4-guidance-map 融合方案：
- 将 4 个 guidance map (depth, normal, semantic, dwpose) 各自重复到 3 通道
- 沿通道维拼接成 12 通道张量
- 用 Conv2d(12→3, kernel_size=1) 投影回 3 通道
- 输入到 ControlNet，完全兼容现有架构

## 架构

```
Depth [T,1,H,W] ──┐
Normal [T,1,H,W] ─┼─→ repeat(1,3,1,1) ──┐
Semantic [T,1,H,W]├────────────────────┼─→ cat(dim=1) ──→ [T,12,H,W]
DWPose [T,1,H,W]─┤                      │
                   └─ repeat(1,3,1,1) ──┘
                              ↓
                   Conv2d(12→3, k=1)
                              ↓
                        [1, 3, H, W] (last frame)
                              ↓
                       prepare_image() 
                              ↓
                        ControlNet
```

## 使用方式

### 方式 1：Python 代码中直接调用

```python
from motion_editor.pipelines import MotionEditorPipeline
import torch

# 加载 pipeline
pipeline = MotionEditorPipeline.from_pretrained("path/to/model")

# 准备 4 个 guidance maps，shape 都是 [T, C, H, W]
# C 可以是 1 或 3，会自动处理
depth_map = torch.randn(8, 1, 512, 512)
normal_map = torch.randn(8, 3, 512, 512)
semantic_map = torch.randn(8, 1, 512, 512)
dwpose_map = torch.randn(8, 3, 512, 512)

# 调用 pipeline，使用 MLMF 模式
with torch.no_grad():
    output = pipeline(
        prompt="a person dancing",
        video_length=8,
        depth_map=depth_map,
        normal_map=normal_map,
        semantic_map=semantic_map,
        dwpose_map=dwpose_map,
        num_inference_steps=50,
        guidance_scale=7.5,
    )

# output.images: 生成的视频帧
```

### 方式 2：从文件加载

```python
# 使用提供的测试脚本
python test_guidance_fusion.py \
    --prompt "a person dancing" \
    --video_length 8 \
    --depth_path ./guidance_maps/depth \
    --normal_path ./guidance_maps/normal \
    --semantic_path ./guidance_maps/semantic \
    --dwpose_path ./guidance_maps/dwpose \
    --output_dir ./outputs
```

### 方式 3：向后兼容的骨架模式

如果不提供 4 个 guidance maps，自动回落到原来的 skeleton 模式：

```python
# Legacy mode - 使用原来的 skeleton
output = pipeline(
    prompt="a person dancing",
    video_length=8,
    skeleton=skeleton_frames,  # 只需要这个
    num_inference_steps=50,
)
```

## 代码改动

### 1. `pipeline_motion_editor.py`

**新增方法**：
- `_fuse_guidance_maps()`: 融合 4 个 guidance maps

**修改内容**：
- 在 `__call__()` 中添加 4 个新参数：`depth_map`, `normal_map`, `semantic_map`, `dwpose_map`
- 在图像准备阶段，检测是否使用 MLMF 或 skeleton 模式
- 自动创建 Conv2d(12→3) 投影层

**关键参数**：
```python
def __call__(
    self,
    ...,
    skeleton=None,  # 原有参数
    depth_map=None,  # 新增
    normal_map=None,  # 新增
    semantic_map=None,  # 新增
    dwpose_map=None,  # 新增
    ...
)
```

### 2. `controlnet_adapter.py`

**无改动**：ControlAdapter 保持原样，不需要修改。Fusion 只在 pipeline 预处理阶段完成。

## 数据格式要求

### 输入格式
所有 guidance maps 都应该是 PyTorch 张量：
- **Shape**: `[T, C, H, W]` 
  - T: 帧数 (通常 8)
  - C: 通道数 (1 或 3)
  - H, W: 空间维度 (通常 512×512)
  
- **Value Range**: `[0, 1]` (float32)

- **通道数处理**：
  - 如果 C=1（单通道），会自动 repeat 到 3 通道
  - 如果 C=3（RGB），直接使用
  - 其他情况会报错

### 转换示例

```python
# 从图像文件加载
from PIL import Image
import torch
import numpy as np

def load_image_sequence(folder_path, num_frames=8):
    """加载一个序列的图像"""
    frames = []
    for i in range(num_frames):
        img = Image.open(f"{folder_path}/frame_{i:04d}.png")
        arr = np.array(img)  # [H, W] or [H, W, 3]
        
        if len(arr.shape) == 2:  # 灰度图
            arr = arr[np.newaxis]  # [1, H, W]
        else:  # RGB 图
            arr = arr.transpose(2, 0, 1)  # [3, H, W]
        
        frames.append(torch.from_numpy(arr).float() / 255.0)
    
    return torch.stack(frames)  # [T, C, H, W]
```

## 投影层的作用

`Conv2d(12→3)` 的作用：
1. **维度匹配**：ControlNet 期望 3 通道输入，我们提供 12 通道
2. **特征融合**：可学习的 1×1 卷积，自动学习如何融合 4 个 guidance 信息
3. **参数量小**：只有 12×3 + 3 = 39 个参数，可以：
   - 随意初始化（现在是随机初始化）
   - 从 CHAMP 预训练权重微调
   - 训练中学习

## 下一步计划

### Phase 2：替换为 CHAMP GuidanceEncoder
- 加载 CHAMP 的 4 个预训练 GuidanceEncoder
- 分别对 4 个 maps 编码
- 融合编码后的 embeddings
- 投影回 3 通道
- 预期效果更好（学到更好的 feature representation）

### Phase 3：多尺度融合
- 在 ControlNet 的多个分辨率层级应用 guidance
- 更充分利用 multi-scale 信息

### Phase 4：可学习融合权重
- 用 attention 机制或其他方式学习 4 个 map 的融合权重
- 自适应地选择哪个 map 更重要

## 调试提示

### 打印调试信息
代码中有调试输出，会打印：
```
[ControlNet] Using multi-layer guidance fusion mode (depth, normal, semantic, dwpose)
images.size(): [2, 1, 3, 512, 512]
```

### 常见错误

1. **"Guidance fusion requires all 4 maps, but X is None"**
   - 原因：使用 MLMF 模式但没有提供某个 map
   - 修复：要么提供全部 4 个 maps，要么不提供，自动回落到 skeleton 模式

2. **"X map has Y channels, expected 1 or 3"**
   - 原因：提供的 map 通道数不是 1 或 3
   - 修复：确保 map shape 是 `[T, 1, H, W]` 或 `[T, 3, H, W]`

3. **显存溢出**
   - 原因：12 通道的中间张量占用更多显存
   - 修复：降低 batch_size，或使用 fp16 混合精度（已配置）

## 性能对比 (预期)

| 指标 | Skeleton | MLMF MVP | MLMF+CHAMP |
|------|---------|----------|-----------|
| 计算时间 | 1x | ~1.05x | ~1.5x |
| 显存用量 | 1x | ~1.02x | ~1.2x |
| 生成质量 | baseline | +10% | +25% |
| 可控性 | 低 | 中 | 高 |

## 参考

- [CHAMP Paper](https://arxiv.org/abs/2403.14781)
- [ControlNet Paper](https://arxiv.org/abs/2302.05543)
- [Diffusers Documentation](https://huggingface.co/docs/diffusers)
