# YOLOv8 红外小目标检测 — HIT-UAV 数据集

> 基于知识蒸馏的轻量级红外无人机小目标检测框架，面向城市灾害搜救场景，以**召回优先**为核心设计目标。

## 项目简介

高空无人机红外小目标检测在搜救中至关重要，但面临双重困难：远距离成像导致目标像素极少、漏检率高，而机载端侧又有实时性硬约束。本文提出一种**召回优先的两阶段教师-学生框架**：先通过贝叶斯调参选定最优教师模型，再通过知识蒸馏将其压缩为轻量级学生模型，在保持召回能力的前提下实现端侧部署。

## 数据集

| 数据集 | 说明 |
|---|---|
| HIT-UAV | 首个高空无人机红外热成像数据集（Scientific Data, 2023），2,898张红外图像，24,899个标注实例，覆盖Person/Car/Bicycle/OtherVehicle四类，飞行高度60-130米 |
| RGBT-DronePerson | 大规模无人机RGB-T行人检测数据集，本文仅取红外分支用于跨域鲁棒性验证 |

> 数据集来源：https://www.kaggle.com/datasets/pandrii000/hituav-a-highaltitude-infrared-thermal-dataset

## 方法流程
五类候选模型基准测试 → TPE贝叶斯调参选教师(YOLOv8x) → 知识蒸馏压缩为学生(YOLOv8n)
### 第一阶段：模型选型

在 HIT-UAV 上对五类模型（YOLOv8n/s/x、RT-DETR-I、Faster R-CNN）进行基准测试，以**小目标召回**为核心指标（而非传统mAP），经 TPE 贝叶斯调参后：

- **教师模型**：YOLOv8x（mAP@0.5 从 0.8142 → 0.9108，FR@10 达 0.9657）
- **学生模型**：YOLOv8n（参数量仅 3.16M，FPS 179.6，满足端侧实时性）

### 第二阶段：知识蒸馏

对比三种蒸馏范式（FGD / CrossKD / ReviewKD），以召回优先为准则选定 **ReviewKD**：

| 方法 | Recall | F2 | 漏帧率 | FR@10 |
|---|---|---|---|---|
| 原始 Student (YOLOv8n) | 0.8422 | 0.8383 | 0.0824 | 0.8979 |
| FGD | 0.9014 | 0.8853 | 0.0647 | 0.9554 |
| CrossKD | 0.9065 | 0.8866 | 0.0706 | 0.9511 |
| **ReviewKD (Ours)** | **0.9082** | **0.8850** | **0.0529** | **0.9571** |

ReviewKD 通过注意力融合模块（ABF）让学生浅层特征逐层回顾教师的多尺度特征，在 Recall 和漏帧率上表现最优。

### 关键发现

- **蒸馏参数**：蒸馏权重 α=0.01、输入分辨率 640 为最优配置；过大 α 会压制学生原本的召回能力
- **跨域分析**：零样本跨域下蒸馏模型存在域敏感倾向（误报激增），但经目标域微调后，ReviewKD 的召回优势重现（FR@10 达 0.7163，为三者最高）

## 核心评价指标

本项目摒弃"mAP中心主义"，采用面向搜救的评价协议：

- **FR@K**：固定误报预算（K=5/10）下的召回率
- **帧级漏检率**：单帧内存在未检出目标的概率
- **F2分数**：Recall 权重是 Precision 的 4 倍

## 项目结构
yolov8-hit-uav/
├── data/
│ └── README.md # 数据集来源与下载方式
├── output/
│ ├── pr_curve.png # PR曲线
│ ├── confusion_matrix.png # 混淆矩阵
│ └── ...
├── notebook/
│ └── train_evaluate.py # 训练+评估代码
├── README.md
└── requirements.txt

## 环境配置

- GPU: NVIDIA GeForce RTX 2080 Ti
- 框架: Ultralytics YOLOv8 + Optuna (TPE)
- 训练: 100 epochs, AdamW, lr=0.01, batch=64
