"""从 checkpoint 重新评估：Macro-F1 / 每类 F1 / 混淆矩阵。

支持 TTA（测试时增强）：原图 + 水平翻转 + 中心放大分别前向，求平均概率。

用法：
  python -m src.evaluate                       # 在 val 集上评估，存 checkpoints/eval_report.json
  python -m src.evaluate --tta flip
  python -m src.evaluate --make-confusion-png  # 额外输出混淆矩阵热力图
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image

from .config import CKPT_DIR, TRAIN_DIR, VAL_DIR, get_model_ckpt
from .train import build_label_map
from .utils.dataset import CarDataset
from .utils.metrics import confusion_matrix, per_class_f1
from .utils.model import build_model
from .inference import _forward_tta, _resolve_img_size, temperature_calibrate


def evaluate_ckpt(
    ckpt_path: Path,
    split_dir: Path,
    device: str = "cpu",
    num_workers: int = 0,
    img_size_override: int = None,
    tta: str = "none",
    temperature: float = 1.0,
) -> dict:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    label_map: Dict[str, int] = ckpt["label_map"]
    classes: List[str] = ckpt["classes"]
    backbone = ckpt.get("extra", {}).get("backbone", "resnet18")
    img_size = int(img_size_override) if img_size_override else _resolve_img_size(ckpt)
    num_classes = len(classes)
    print(f"使用推理尺寸: {img_size}px, TTA: {tta}, T={temperature}")

    model = build_model(backbone, num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    ds = CarDataset(split_dir, label_map, transform=None)
    print(f"评估集: {split_dir}  样本数: {len(ds)}  类别数: {num_classes}")

    y_true, y_pred = [], []
    with torch.no_grad():
        for i in range(len(ds)):
            img_pil, label, _ = ds[i]
            img_pil = img_pil.convert("RGB") if hasattr(img_pil, "convert") else Image.open(img_pil).convert("RGB")
            probs = _forward_tta(model, img_pil, img_size, tta=tta).cpu()
            if temperature and abs(temperature - 1.0) > 1e-6:
                probs = temperature_calibrate(probs, T=temperature)
            y_pred.append(int(probs.argmax()))
            y_true.append(int(label))
            if (i + 1) % 200 == 0:
                print(f"  已评估 {i + 1}/{len(ds)}")

    f1s = per_class_f1(y_true, y_pred, num_classes)
    cm = confusion_matrix(y_true, y_pred, num_classes)
    result = {
        "checkpoint": str(ckpt_path),
        "eval_dir": str(split_dir),
        "num_samples": len(ds),
        "num_classes": num_classes,
        "img_size": img_size,
        "tta": tta,
        "temperature": temperature,
        "macro_f1": float(f1s.mean()),
        "acc": float(np.mean(np.array(y_true) == np.array(y_pred))),
        "classes": classes,
        "per_class_f1": {classes[i]: float(f1s[i]) for i in range(num_classes)},
        "confusion_matrix": cm.tolist(),
    }
    return result


def draw_confusion_png(classes: List[str], cm: np.ndarray, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(cm, cmap="Blues", ax=ax, cbar=True)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (51 classes)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"混淆矩阵已保存 {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=None)
    p.add_argument("--split", default="val", choices=["val", "train"])
    p.add_argument("--out", default=None)
    p.add_argument("--img-size", type=int, default=None, help="覆盖推理尺寸")
    p.add_argument("--tta", default="flip",
                   choices=["none", "flip", "flip+scale", "5crop", "5crop+flip"],
                   help="测试时增强策略（默认 flip，本数据集上最优）")
    p.add_argument("--temperature", type=float, default=1.0, help="置信度温度缩放")
    p.add_argument("--make-confusion-png", action="store_true")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    ckpt_path = Path(args.ckpt) if args.ckpt else get_model_ckpt()
    split_dir = VAL_DIR if args.split == "val" else TRAIN_DIR
    out = Path(args.out) if args.out else (CKPT_DIR / "eval_report.json")

    res = evaluate_ckpt(
        ckpt_path, split_dir, device=args.device,
        img_size_override=args.img_size, tta=args.tta,
        temperature=args.temperature,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"\n=== 评估结果 ===")
    print(f"Macro-F1 = {res['macro_f1']:.4f}")
    print(f"Acc      = {res['acc']:.4f}")
    print(f"unknown F1 = {res['per_class_f1'].get('unknown', 0):.4f}")
    print(f"已存 {out}")

    if args.make_confusion_png:
        cm = np.array(res["confusion_matrix"])
        draw_confusion_png(res["classes"], cm, CKPT_DIR / "confusion_matrix.png")


if __name__ == "__main__":
    main()