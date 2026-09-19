"""批量推理：对指定测试目录里的每张图出一行 predictions.csv。

格式（任务书 8 节）：
  image_id,predicted_label,confidence
  - predicted_label 是 4 位 ID（带前导零）或 "unknown"
  - confidence 0~1
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List

from .config import get_model_ckpt, get_reject_threshold
from .inference import load_model, _forward_tta, temperature_calibrate, apply_unknown_fallback
from .utils.dataset import IMG_EXTS

import torch
from PIL import Image


@torch.no_grad()
def batch_predict(test_dir: Path, ckpt_path: Path, out_csv: Path,
                  device: str = "cpu", tta: str = "flip",
                  temperature: float = 1.0, reject_tau: float = None) -> List[dict]:
    test_dir = Path(test_dir)
    if not test_dir.is_dir():
        raise SystemExit(f"找不到测试目录: {test_dir}")
    if reject_tau is None:
        reject_tau = get_reject_threshold()

    # 收集所有图片（递归 + 非递归都覆盖）
    files = sorted([p for p in test_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS])
    if not files:
        raise SystemExit(f"测试目录里没找到任何图片：{test_dir}")

    model, classes, label_map, img_size = load_model(ckpt_path, device=device)
    rows: List[dict] = []

    for i, p in enumerate(files, 1):
        try:
            img = Image.open(p).convert("RGB")
            probs = _forward_tta(model, img, img_size, tta=tta).cpu()
            if temperature and abs(temperature - 1.0) > 1e-6:
                probs = temperature_calibrate(probs, T=temperature)
            # unknown 拒识回退（阈值在验证集标定）
            top1_id, top1_conf, _ = apply_unknown_fallback(probs, classes, tau=reject_tau)
            rows.append({"image_id": p.name, "predicted_label": top1_id,
                         "confidence": round(float(top1_conf), 6)})
        except Exception as e:
            print(f"  ! 跳过 {p.name}: {e}", file=sys.stderr)
        if i % 200 == 0:
            print(f"已处理 {i}/{len(files)} 张")

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "predicted_label", "confidence"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"写出 {out_csv}：{len(rows)} 条记录  (TTA={tta}, T={temperature}, tau={reject_tau})")
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--test-dir", required=True)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--out", default="predictions.csv")
    p.add_argument("--device", default="cpu")
    p.add_argument("--tta", default="flip",
                   choices=["none", "flip", "flip+scale", "5crop", "5crop+flip"])
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--reject-tau", type=float, default=None,
                   help="unknown 拒识回退阈值（默认取 .env / config，标定值 0.35）")
    args = p.parse_args()
    batch_predict(
        test_dir=Path(args.test_dir),
        ckpt_path=Path(args.ckpt) if args.ckpt else get_model_ckpt(),
        out_csv=Path(args.out),
        device=args.device,
        tta=args.tta,
        temperature=args.temperature,
        reject_tau=args.reject_tau,
    )


if __name__ == "__main__":
    main()
