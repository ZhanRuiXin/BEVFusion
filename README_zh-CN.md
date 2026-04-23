# BEVFusion 时序多帧融合项目

基于 PyTorch 1.10.2 + CUDA 11.3 的 BEVFusion 时序多帧融合环境

## 📖 项目简介

本项目实现基于 BEVFusion 的时序多帧融合算法，通过融合多帧点云和图像信息，提升 3D 目标检测的准确性和稳定性。

### 核心特性

- **时序多帧融合**：利用历史帧信息增强当前帧的感知能力
- **BEV 特征对齐**：通过时序对齐模块融合多帧 BEV 特征
- **端到端训练**：支持点云和图像的多模态端到端训练
- **高效推理**：优化的时序缓存机制，支持实时推理

### 技术栈

| 组件 | 版本 |
|------|------|
| 基础系统 | Ubuntu 20.04 |
| CUDA | 11.3 |
| cuDNN | 8 |
| PyTorch | 1.11.0 |
| Python | 3.8 |
| MMDetection3D | 1.x |
| MMEngine | 0.10.0 |
| MMCV | 2.0.0 |

## 🚀 快速开始

### 前置要求

- Docker Engine 20.10+
- NVIDIA Container Toolkit（用于 GPU 支持）
- NVIDIA GPU 驱动版本 >= 455.23

### 构建 Docker 镜像

`docker build -t bevfusion:latest docker/`

### 运行容器

基础运行（使用 GPU 支持）：

`docker run --gpus all --shm-size=16g -it bevfusion:latest`

挂载本地项目目录：

`docker run --gpus all --shm-size=16g -v /path/to/your/project:/workspace -it bevfusion:latest`

带端口映射（用于可视化等）：

`docker run --gpus all --shm-size=16g -p 8888:8888 -p 6006:6006 -v /path/to/your/project:/workspace -it bevfusion:latest`

### VS Code Dev Container 使用

创建 `.devcontainer/devcontainer.json`：

```
{
    "name": "BEVFusion",
    "image": "bevfusion:latest",
    "runArgs": ["--gpus", "all", "--shm-size=16g"],
    "customizations": {
        "vscode": {
            "extensions": [
                "ms-python.python",
                "ms-toolsai.jupyter"
            ]
        }
    }
}
```

### 验证环境

进入容器后，运行以下命令验证安装：

检查 Python 环境：

`python --version`

检查 PyTorch 和 CUDA：

`python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda)"`

检查 MMDetection3D：

`python -c "import mmdet3d; print('MMDetection3D:', mmdet3d.__version__)"`

## 📦 时序多帧融合架构

### 整体流程

1. **多模态特征提取**
   - 点云分支：使用 PointPillars 或 VoxelNet 提取点云特征
   - 图像分支：使用 ResNet 或 Swin Transformer 提取图像特征

2. **BEV 特征转换**
   - 点云 BEV：通过 Pillar Scatter 或 Voxel Pooling 转换
   - 图像 BEV：通过 LSS (Lift-Splat-Shoot) 或 Transformer 转换

3. **时序融合模块**
   - 历史帧缓存：维护固定长度的历史 BEV 特征队列
   - 时序对齐：根据位姿信息对齐多帧 BEV 特征
   - 特征融合：使用 ConvLSTM 或注意力机制融合多帧特征

4. **检测头**
   - 3D 目标检测：CenterPoint 或 TransFusion 头
   - 时序一致性优化：添加时序损失函数

### 关键参数配置

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| 历史帧数 | 3-5 | 时序融合使用的历史帧数量 |
| BEV 尺寸 | 200x200 | BEV 特征图的尺寸 |
| 体素尺寸 | (0.2, 0.2, 8) | 点云体素化参数 |
| 时序对齐 | 精确 | 是否使用精确的位姿对齐 |

## 🔧 训练与评估

### 数据准备

数据集目录结构应如下：

```
data/
├── nuscenes/
│   ├── samples/
│   ├── sweeps/
│   ├── maps/
│   └── v1.0-trainval/
├── kitti/
│   ├── training/
│   └── testing/
└── waymo/
    ├── training/
    └── validation/
```

### 单卡训练

`python tools/train.py configs/bevfusion/bevfusion_temporal.py --work-dir work_dirs/bevfusion_temporal`

### 多卡训练

```
bash tools/dist_train.sh configs/bevfusion/bevfusion_temporal.py 4 --work-dir work_dirs/bevfusion_temporal
```

### 评估模型

```
python tools/test.py configs/bevfusion/bevfusion_temporal.py /path/to/checkpoint.pth --eval mAP
```

### 可视化结果

```python
 tools/analysis_tools/visualize.py configs/bevfusion/bevfusion_temporal.py --result /path/to/result.pkl --show
```

## 📝 配置文件示例

### bevfusion_temporal.py 关键配置

时序融合配置
```
temporal = dict(
    enabled=True,
    history_frames=3,
    alignment='pose',  # 'pose' or 'flow'
    fusion_method='convlstm',  # 'concat', 'weighted', 'convlstm', 'attention'
    cache_size=10,
)
```

# BEV 配置
```
bev = dict(
    x_range=[-51.2, 51.2],
    y_range=[-51.2, 51.2],
    x_size=200,
    y_size=200,
)
```

# 模型配置
```
model = dict(
    type='BEVFusionTemporal',
    temporal_cfg=temporal,
    bev_cfg=bev,
    point_cloud_range=point_cloud_range,
)
```