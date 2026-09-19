"""训练入口：51 类（50 + unknown）。

策略：
- 选 ResNet50 / ResNet18 / EfficientNet-B0 / ConvNeXt-Tiny 之一做迁移学习。
- 用验证集 Macro-F1 选择 best checkpoint（与训练目标一致）。
- unknown 当成"第 51 类"普通分类（任务书允许的同时，明确要求"不能用置信度阈值
  替代 unknown 类的训练"）。
- 自动设备：如果有 CUDA 用 CUDA，否则 CPU。
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T

from .config import CKPT_DIR, TRAIN_DIR, VAL_DIR, get_model_ckpt
from .utils.dataset import CarDataset, load_class_index
from .utils.metrics import confusion_matrix, macro_f1, per_class_f1
from .utils.model import (
    DEFAULT_IMG_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_model,
    save_checkpoint,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mixup_batch(imgs: torch.Tensor, labels: torch.Tensor, alpha: float):
    """MixUp（Zhang et al. 2018）。

    按 Beta(alpha, alpha) 抽一个 lambda，把 batch 内的样本两两线性混合；
    损失按 lambda 加权到两组目标上。只在训练时启用，验证集照常做常规评估，
    因此指标口径与不启用时保持一致，可以直接对比。
    """
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(imgs.size(0), device=imgs.device)
    return lam * imgs + (1.0 - lam) * imgs[idx], labels, labels[idx], lam


def make_transforms(img_size: int, train: bool) -> T.Compose:
    if train:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_label_map(classes_file: Path, include_unknown: bool = True) -> Dict[str, int]:
    ids = [l.strip() for l in Path(classes_file).read_text(encoding="utf-8").splitlines() if l.strip()]
    if len(ids) != 50:
        raise ValueError(f"classes.txt 应为 50 行，实际 {len(ids)}")
    m = load_class_index(ids)
    if include_unknown:
        m["unknown"] = len(m)
    return m


def evaluate(model, loader, device, num_classes: int) -> dict:
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            imgs, labels, _ = batch
            imgs = imgs.to(device)
            logits = model(imgs)
            preds = logits.argmax(dim=1).cpu().numpy().tolist()
            y_pred.extend(preds)
            y_true.extend(labels.numpy().tolist())
    f1s = per_class_f1(y_true, y_pred, num_classes)
    mf1 = float(f1s.mean())
    acc = float(np.mean(np.array(y_true) == np.array(y_pred)))
    cm = confusion_matrix(y_true, y_pred, num_classes)
    return {"macro_f1": mf1, "acc": acc, "per_class_f1": f1s, "cm": cm}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--classes", default="classes.txt")
    p.add_argument("--backbone", default="resnet18", choices=["resnet50", "resnet18", "efficientnet_b0", "convnext_tiny"])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-best", default=None, help="best 模型保存路径，默认 checkpoints/best.pt")
    p.add_argument("--limit-per-class", type=int, default=None, help="调试用，限制每个类最多几张图")
    p.add_argument("--mixup-alpha", type=float, default=0.0, help="MixUp 的 Beta 分布参数；0 表示关闭")
    p.add_argument("--cosine", action="store_true", help="启用余弦退火学习率（lr 线性衰减到 lr-min）")
    p.add_argument("--lr-min", type=float, default=1e-5, help="余弦退火的终点学习率")
    args = p.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    classes_file = Path(args.classes)
    if not classes_file.exists():
        raise SystemExit(f"找不到 {classes_file}，请先跑 src.data_prep make-classes")
    label_map = build_label_map(classes_file, include_unknown=True)
    classes = list(label_map.keys())
    print(f"类别: {len(classes)}（含 unknown）")
    num_classes = len(classes)

    train_ds = CarDataset(TRAIN_DIR, label_map, transform=make_transforms(args.img_size, True), limit_per_class=args.limit_per_class)
    val_ds = CarDataset(VAL_DIR, label_map, transform=make_transforms(args.img_size, False), limit_per_class=args.limit_per_class)
    print(f"训练样本: {len(train_ds)}, 验证样本: {len(val_ds)}")
    if len(train_ds) == 0:
        raise SystemExit("训练集为空，请先跑 src.data_prep make-unknown")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model(args.backbone, num_classes=num_classes, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # 余弦退火：让后期的学习率逐步趋近 lr_min，避免末段在最优解附近来回震荡
    scheduler = None
    if args.cosine:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr_min
        )
        print(f"scheduler: CosineAnnealingLR, {args.lr:.1e} -> {args.lr_min:.1e}")
    if args.mixup_alpha > 0:
        print(f"mixup: 已启用, alpha={args.mixup_alpha}")
    print(f"计划: {args.epochs} 个 epoch, batch={args.batch_size}, img={args.img_size}")

    save_path = Path(args.save_best) if args.save_best else get_model_ckpt()
    save_path.parent.mkdir(parents=True, exist_ok=True)

    best_f1 = -1.0
    history: List[dict] = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        loss_sum, n = 0.0, 0
        for batch in train_loader:
            imgs, labels, _ = batch
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            if args.mixup_alpha > 0:
                imgs, y_a, y_b, lam = mixup_batch(imgs, labels, args.mixup_alpha)
                logits = model(imgs)
                loss = lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
            else:
                logits = model(imgs)
                loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * labels.size(0)
            n += labels.size(0)
        train_loss = loss_sum / max(1, n)
        if scheduler is not None:
            scheduler.step()
        eval_res = evaluate(model, val_loader, device, num_classes)
        elapsed = time.time() - t0
        cur_lr = float(optimizer.param_groups[0]["lr"])
        line = (
            f"[epoch {epoch:02d}/{args.epochs}] lr={cur_lr:.2e} loss={train_loss:.4f} "
            f"val_macro_f1={eval_res['macro_f1']:.4f} val_acc={eval_res['acc']:.4f} time={elapsed:.1f}s"
        )
        print(line, flush=True)
        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "macro_f1": float(eval_res["macro_f1"]),
            "acc": float(eval_res["acc"]),
            "per_class_f1": [float(x) for x in eval_res["per_class_f1"]],
            "time_sec": float(elapsed),
            "backbone": args.backbone,
            "lr": cur_lr,
            "mixup_alpha": float(args.mixup_alpha),
        })
        if eval_res["macro_f1"] > best_f1:
            best_f1 = eval_res["macro_f1"]
            save_checkpoint(
                save_path,
                model,
                label_map=label_map,
                classes=classes,
                img_size=args.img_size,
                extra={
                    "epoch": epoch,
                    "macro_f1": best_f1,
                    "backbone": args.backbone,
                    "img_size": args.img_size,
                    "train_img_size": args.img_size,
                    # 推理尺寸沿用 v1 的实验结论（224 + flip TTA 最优），见 report 4.2
                    "infer_img_size": 224,
                    "mixup_alpha": float(args.mixup_alpha),
                    "cosine": bool(args.cosine),
                    "lr": float(args.lr),
                    "train_samples": len(train_ds),
                },
            )
            # 同时把 best 的评估详情单独落盘（报告 4.2 节需要混淆矩阵 / 每类 F1）
            with open(CKPT_DIR / "best_eval.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "epoch": epoch,
                        "macro_f1": float(eval_res["macro_f1"]),
                        "acc": float(eval_res["acc"]),
                        "per_class_f1": [float(x) for x in eval_res["per_class_f1"]],
                        "classes": classes,
                        "confusion_matrix": eval_res["cm"].tolist(),
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            print(f"  ↳ best updated, saved {save_path}")

    # history + 最佳指标落盘
    with open(CKPT_DIR / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f"训练完成。best Macro-F1 = {best_f1:.4f}")


if __name__ == "__main__":
    main()
