# Multi-Breed Dog 2D Pose Estimation with ViTPose++

## Overview
This project develops a machine learning pipeline for accurate 2D skeletal pose estimation across multiple dog breeds using synthetic training data and Vision Transformer-based architectures.

## Motivation
Automated dog pose estimation has applications in:
- **Veterinary biomechanics**: Gait analysis, injury detection
- **Computer vision**: Animal tracking, behavior analysis
- **Robotics**: Canine-robot interaction
- **Sports**: Agility competition monitoring

Traditional approaches struggle with breed diversity and occlusion. This work leverages synthetic data to solve domain-specific challenges.

## Technical Approach

### Architecture
- **Backbone**: ViTPose++ (Vision Transformer, COCO-pretrained)
  - Merges AP10K animal expert from MoE layers
  - Processes 256×192 input images
  - Outputs 64×48 heatmaps for 49 dog keypoints

- **Head**: Deconvolutional upsampler (4× stride-4 upsample)
  - Maps ViT patch tokens to spatial heatmaps
  - Final output: [B, 49, 64, 48]

- **Decoding**: Soft-argmax (differentiable peak detection)
  - Extracts joint coordinates from heatmaps
  - Sub-pixel accuracy via softmax-weighted accumulation

### Loss Function
**Masked Heatmap Loss**:
- Per-joint MSE on heatmaps, weighted by visibility
- Visibility = 0: 10% penalty (regularization)
- Visibility = 1: full penalty (supervised)
- Normalized by count of visible joints per batch

### Metric
**Percentage of Correct Keypoints (PCK)**:
- Joint within threshold × bbox diagonal of GT
- Threshold: 0.1 (standard for pose estimation)
- Only evaluated on visible joints

## Dataset

### Synthetic Data Generation
- **Blender-rendered**: 4.2M images across 12 dog breeds
- **Multi-camera**: 5 viewpoints (rightside, leftside, top, headside, backside)
- **Multi-focal**: 9 focal lengths (100-200mm equivalent)
- **Multi-action**: 20+ actions (walk, sit, jump, etc.)
- **Textures**: AO, Albedo, Normal variants

### Keypoint Annotation
- **49-bone skeleton**: Anatomically accurate
  - Limbs: forelimbs, hindlimbs (with claws, digits)
  - Spine: 6 joints (base to T3)
  - Head: eyes, ears, nose, mouth, tongue
  - Tail: 4 segments
  - Helper bones: virtual anatomy guides

### Label Space
- Coordinates in **1920×1080** (Blender label space)
- Scaled to **960×540** (rendered image space) at load time
- Normalized to [0,1] per input image

## Data Pipeline

### Preprocessing
1. **Index Construction** (breed, action)-level split
   - Groups samples by (breed, action) pairs
   - Ensures breed-action combinations don't leak between splits
   - Supports 80/10/10 train/val/test split

2. **Coordinate Scaling**
   - Label space (1920×1080) → image space (960×540)
   - Image space → input space (256×192) via crop + resize

3. **Camera-Aware Visibility Masking**
   - Occlusion encoding per camera angle
   - Camera_rightside: sees right-side joints only
   - Camera_leftside: sees left-side joints only
   - Camera_top: occludes feet (ground contact)
   - Enables realistic supervision

4. **Augmentation**
   - Random bbox jitter (±10% center, ±10% scale)
   - Random horizontal flip (50%)
   - Color jitter (brightness, contrast, saturation)
   - ImageNet normalization

### Loading
- **Pickle Index Files**:
  - 3 pkl files: index_train.pkl, index_val.pkl, index_test.pkl
  - Stores pre-computed keypoints + metadata
  - 4.2M samples, ~3GB total
  - Loaded once at dataset init

- **Efficient I/O**:
  - Multi-worker DataLoader with pinned memory
  - Breed-interleaved sampling (spreads NFS load)
  - Parallel CSV parsing during index build

## Training Configuration

### Hyperparameters
- **Optimizer**: AdamW (lr=1e-4, weight_decay=1e-4)
- **Scheduler**: Warmup LinearLR + CosineAnnealingLR
- **Batch Size**: 128 (tunable)
- **Epochs**: 10+ (early stopping on val PCK)
- **AMP**: Mixed precision training (if GPU available)

### Hardware
- **GPU**: NVIDIA A100/V100/RTX6000 (batch_size=128)
- **SLURM Cluster**: Pegasus (Kaiserslautern)
- **Container**: Singularity/Apptainer with PyTorch 

## Results & Metrics

### Expected Performance
- **Synthetic Val PCK**: ~99% (near-perfect on rendered data)
- **Real-World PCK**: TBD (domain gap under investigation)
  - Synthetic→real gap likely due to:
    - Rendering artifacts
    - Breed variation not fully covered
    - Texture/lighting domain shift

### Visualization
- **Keypoint overlays**: Predicted (blue) vs GT (green) on images
- **Heatmap grids**: All 49 target + predicted heatmaps (7×7 grid)
   
## Project Structure
```
vit3/
├── dog_pose_dataset.py      # Data loading, preprocessing, index building
├── model.py                 # ViTPose++ architecture, loss, metrics
├── train.py                 # Training loop, optimization, visualization
├── config.py                # Configuration (geometry, augmentation, training)
├── bone_order.json          # 49-bone skeleton definition
├── train_visualizations/    # Output keypoint + heatmap visualizations
└── checkpoints/             # Saved model weights

```

## Key Innovations
1. **Camera-aware visibility masking**: Realistic occlusion supervision
2. **(Breed, action)-level splitting**: Prevents breed bias in validation
3. **Synthetic-to-real benchmark**: 4.2M renders across 12 breeds
4. **ViTPose++ MoE integration**: Leverages AP10K animal expert
5. **Efficient multi-worker I/O**: Breed-interleaved sampling for NFS parallelism

## Future Work
- Real-world dataset collection + domain adaptation
- Multi-view 3D pose lifting
- Temporal consistency (video-based refinement)
- Occlusion-robust inference
- Breed-specific fine-tuning

## References
- **ViTPose**: Xu et al., "ViTPose: Simple Vision Transformer Baselines for Human Pose Estimation" (CVPR 2023)
- **ViTPose++**: Xu et al., "ViTPose++: Vision Transformer for Generic Body Pose Estimation" (CVPR 2024)
- **AP10K**: Yu et al., "Towards General Animal Pose and Shape Estimation via Weakly-Supervised Learning" (CVPR 2021)
- **SMPL**: Loper et al., "SMPL: A Skinned Multi-Person Linear Model" (TOG 2015)

## Contact & Citation
**Author**: Alcina (DFKI, Kaiserslautern)  
**Status**: Active Development 

---
*This project is part of research into parametric dog anatomy modeling and motion capture retargeting at DFKI Embedded Systems group.*
