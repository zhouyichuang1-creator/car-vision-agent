"""模型定义：基于 torchvision 的迁移学习 baseline。

任务：51 类（50 车型 + 1 unknown）。模型 index 0..49 对应 classes.txt 顺序，
index 50 是 unknown。训练时把 unknown 当作一个普通类别直接参与训练，
不靠置信度阈值兜底；阈值仅用于推理阶段的额外拒识（任务书允许，但不能用它替代
unknown 类的训练）。
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
from torchvision import models

# 可选 backbone
_BACKBONES = {
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1),
    "convnext_tiny": (models.convnext_tiny, models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1),
}


def build_model(backbone: str = "resnet50", num_classes: int = 51, pretrained: bool = True) -> nn.Module:
    if backbone not in _BACKBONES:
        raise ValueError(f"unknown backbone: {backbone}. choices={list(_BACKBONES)}")
    ctor, wcls = _BACKBONES[backbone]
    weights = wcls if pretrained else None
    model = ctor(weights=weights)
    # 替换最后一层
    if backbone.startswith("resnet"):
        in_feat = model.fc.in_features
        model.fc = nn.Linear(in_feat, num_classes)
    elif backbone.startswith("efficientnet"):
        in_feat = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_feat, num_classes)
    elif backbone.startswith("convnext"):
        in_feat = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_feat, num_classes)
    else:
        raise ValueError(backbone)
    return model


# 标准 ImageNet 预处理（224x224）
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_IMG_SIZE = 224


def save_checkpoint(
    path,
    model: nn.Module,
    label_map: Dict[str, int],
    classes: list,
    img_size: int = DEFAULT_IMG_SIZE,
    extra: Optional[dict] = None,
) -> None:
    """保存模型权重 + 标签映射 + 预处理参数（任务书要求"模型结构、类别映射、预处理参数、训练配置"一并保存）。"""
    state = {
        "state_dict": model.state_dict(),
        "label_map": label_map,
        "classes": classes,
        "img_size": int(img_size),  # 顶层就是真实训练尺寸，推理直接读这里
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "extra": extra or {},
    }
    torch.save(state, path)


def load_checkpoint(path, backbone: str = "resnet50", num_classes: int = 51, device: str = "cpu") -> dict:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    # 兼容"轻量加载"：不强制重建模型，让调用方决定怎么 load_state_dict
    return ckpt
