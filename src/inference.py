"""加载训练好的 checkpoint，给出"单张图推理"。
输出：classes.txt 中的一位 ID，或字符串 "unknown"，及 confidence（0~1）。

支持测试时增强（TTA）：原图 + 水平翻转，分别前向，取平均概率。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from torchvision import transforms as T

from .config import CLASS_INFO_JSON, get_model_ckpt, get_reject_threshold, get_top_k
from .utils.model import IMAGENET_MEAN, IMAGENET_STD, build_model


def _resolve_img_size(ckpt: dict) -> int:
    """决定推理预处理尺寸。

    优先级：
      1. checkpoint extra.infer_img_size（实验选定的推理尺寸，报告里会说明）
      2. checkpoint 顶层 img_size（训练时的真实尺寸）
      3. 224 兜底
    """
    extra = ckpt.get("extra") or {}
    if isinstance(extra, dict) and extra.get("infer_img_size"):
        return int(extra["infer_img_size"])
    return int(ckpt.get("img_size", 224))


def load_model(ckpt_path: Path, device: str = "cpu"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["state_dict"]
    label_map = ckpt["label_map"]  # id -> idx
    classes = ckpt["classes"]      # idx -> id
    img_size = _resolve_img_size(ckpt)
    backbone = ckpt.get("extra", {}).get("backbone", "resnet18")
    model = build_model(backbone, num_classes=len(classes), pretrained=False)
    model.load_state_dict(state)
    model.eval()
    return model, classes, label_map, img_size


def preprocess(img: Image.Image, img_size: int) -> torch.Tensor:
    tf = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return tf(img)


def softmax(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, dim=-1)


@torch.no_grad()
def _forward_tta(model, img_rgb: Image.Image, img_size: int, tta: str = "flip"):
    """对单张 RGB 图做 TTA 前向，返回平均后的 softmax 概率（CPU tensor）。

    tta 取值：
      - "none": 仅原图
      - "flip": 原图 + 水平翻转（2 路平均）
      - "flip+scale": 原图 + 水平翻转 + 中心放大 1.1 倍再缩回（3 路平均）
      - "5crop": 中心 + 四角 + 各自水平翻转（10 路平均）——最贵但最准
      - "5crop+flip": 同上（默认 = 最稳）
    """
    x0 = preprocess(img_rgb, img_size).unsqueeze(0)
    probs = softmax(model(x0)).squeeze(0)
    if tta == "none":
        return probs

    if tta in ("flip", "flip+scale"):
        x_flip = preprocess(T.functional.hflip(img_rgb), img_size).unsqueeze(0)
        probs = (probs + softmax(model(x_flip)).squeeze(0)) / 2

    if "scale" in tta:
        # 中心放大 1.1 再缩回
        from torchvision.transforms.functional import resize
        w, h = img_rgb.size
        scaled = resize(img_rgb, [int(h * 1.1), int(w * 1.1)])
        scaled_back = resize(scaled, [h, w])
        x_sc = preprocess(scaled_back, img_size).unsqueeze(0)
        probs = (probs + softmax(model(x_sc)).squeeze(0)) / 2
    elif tta in ("5crop", "5crop+flip"):
        # 五裁剪：左上/右上/左下/右下/中心 + 各自的水平翻转（共 10 路）
        w, h = img_rgb.size
        crop_size = int(min(w, h) * 0.875)  # 取 87.5% 中心区
        if crop_size < img_size:
            crop_size = min(w, h)
        # 5 个裁剪起点（左上角坐标）
        cx0 = (w - crop_size) // 2
        cy0 = (h - crop_size) // 2
        coords = [
            (0, 0),
            (w - crop_size, 0),
            (0, h - crop_size),
            (w - crop_size, h - crop_size),
            (cx0, cy0),
        ]
        crops = []
        for x, y in coords:
            box = (x, y, x + crop_size, y + crop_size)
            crops.append(img_rgb.crop(box))
        total = 1  # 原图已计
        for c in crops:
            xc = preprocess(c, img_size).unsqueeze(0)
            probs = probs + softmax(model(xc)).squeeze(0)
            total += 1
            if tta == "5crop+flip":
                xf = preprocess(T.functional.hflip(c), img_size).unsqueeze(0)
                probs = probs + softmax(model(xf)).squeeze(0)
                total += 1
        probs = probs / total
    return probs


def temperature_calibrate(probs: torch.Tensor, T: float = 1.4) -> torch.Tensor:
    """温度缩放：用 T>1 软化 softmax，让 conf 数值更"诚实"。

    训练时模型过拟合后 softmax 容易偏尖——一张陌生图也可能输出 0.99，
    让用户误以为"非常可信"。T>1 会把概率摊开一点，让低把握的图 conf 显低。

    注意：这一步只影响**显示的置信度**，不影响 top1_id（top1 排序不变）。
    """
    if abs(T - 1.0) < 1e-6:
        return probs
    # softmax(logit/T) ∝ probs^(1/T)
    # 用 power 形式等价避免再前向
    p = probs.clamp_min(1e-9)
    p = p.pow(1.0 / T)
    return p / p.sum()


def apply_unknown_fallback(
    probs: torch.Tensor,
    classes: List[str],
    tau: float = 0.0,
) -> Tuple[str, float, List[Tuple[str, float]]]:
    """unknown 拒识回退规则（阈值 tau 在验证集上标定，任务书要求）。

    背景：训练集里 unknown 被过采样，模型容易过度拒识——已知类图片也会被
    判成 unknown。本规则在**不重训**的前提下做一次事后校正：

      - 若 51 类 argmax 是某已知类 -> 直接采用
      - 若 argmax 是 unknown，但"最佳已知类"的概率 >= tau -> 改判为该已知类
      - 否则保留 unknown

    返回 (top1_id, top1_conf, 按概率降序的 [(id, prob), ...])。
    """
    # 先按概率降序排好，保证 top_k 语义正确
    items = sorted(((classes[i], float(probs[i])) for i in range(len(classes))),
                   key=lambda x: -x[1])
    top1_id, top1_conf = items[0]

    if top1_id == "unknown" and tau > 0:
        bk_id, bk_p = next((c, p) for c, p in items if c != "unknown")
        if bk_p >= tau:
            top1_id, top1_conf = bk_id, bk_p
            # 把被提升的已知类排到最前，保持 top_k 与 top1 一致
            items = [(bk_id, bk_p)] + [(c, p) for c, p in items if c != bk_id]
    return top1_id, top1_conf, items


@torch.no_grad()
def predict(
    image_path: Path,
    ckpt_path: Path = None,
    top_k: int = 5,
    reject_threshold: float = 0.0,
    device: str = "cpu",
    img_size: int = None,
    tta: str = "flip",
    temperature: float = 1.0,
    reject_tau: float = None,
) -> dict:
    """主入口。返回 {top1_id, top1_conf, top_k: [(id, conf), ...], img_size_used, tta}。

    img_size 为 None 时按 checkpoint 内记录的训练尺寸；显式传入则覆盖（用于实验对比）。
    tta: "none" / "flip" / "flip+scale" / "5crop" / "5crop+flip"，默认 "flip"（在我们数据上最优）。
    temperature: 置信度校准温度。1.0=原始；>1 会让 conf 数值更"诚实"（陌生图 conf 会偏低）。
    reject_tau: unknown 拒识回退阈值（验证集标定）。None 时取 config 中的默认值；<=0 表示关闭。
    """
    if ckpt_path is None:
        ckpt_path = get_model_ckpt()
    if reject_tau is None:
        reject_tau = get_reject_threshold()
    image_path = Path(image_path)
    model, classes, label_map, ckpt_img_size = load_model(ckpt_path, device=device)
    use_size = int(img_size) if img_size else ckpt_img_size
    img = Image.open(image_path).convert("RGB")
    probs = _forward_tta(model, img, use_size, tta=tta).cpu()
    if temperature and abs(temperature - 1.0) > 1e-6:
        probs = temperature_calibrate(probs, T=temperature)

    # unknown 拒识回退（tau 在验证集标定；tau<=0 表示不启用）
    if reject_tau and reject_tau > 0:
        top1_id, top1_conf, items = apply_unknown_fallback(probs, classes, tau=reject_tau)
        top_k_list = items[:min(top_k, len(items))]
    else:
        topk = min(top_k, len(classes))
        vals, idxs = torch.topk(probs, topk)
        top1_idx = int(idxs[0])
        top1_id = classes[top1_idx]
        top1_conf = float(vals[0])
        top_k_list = [(classes[int(i)], float(v)) for v, i in zip(vals, idxs)]

    out = {
        "image_id": image_path.name,
        "top1_id": top1_id,
        "top1_conf": top1_conf,
        "img_size_used": use_size,
        "tta": tta,
        "temperature": temperature,
        "reject_tau": reject_tau,
        "top_k": top_k_list,
    }
    return out


def cmd_single(args: argparse.Namespace) -> None:
    res = predict(
        image_path=Path(args.image),
        ckpt_path=Path(args.ckpt) if args.ckpt else None,
        top_k=args.top_k,
        reject_threshold=args.reject,
        tta=args.tta,
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))


def cmd_bench_tta(args: argparse.Namespace) -> None:
    """对比 tta=none / flip / flip+scale 的差异，方便报告里给出实验数据。"""
    from .utils.dataset import IMG_EXTS
    ckpt_path = Path(args.ckpt) if args.ckpt else get_model_ckpt()
    test_dir = Path(args.test_dir)
    files = sorted([p for p in test_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS])[:args.limit]
    print(f"benchmark on {len(files)} files, ckpt={ckpt_path}")
    model, classes, label_map, ckpt_img_size = load_model(ckpt_path)
    img_size = args.img_size or ckpt_img_size
    rows = []
    for p in files:
        img = Image.open(p).convert("RGB")
        for tag, tta in [("none", "none"), ("flip", "flip"), ("flip+scale", "flip+scale")]:
            probs = _forward_tta(model, img, img_size, tta=tta).cpu()
            v, i = torch.max(probs, dim=0)
            rows.append({"image": p.name, "tta": tag, "top1_id": classes[int(i)], "conf": round(float(v), 6)})
    print(json.dumps(rows, ensure_ascii=False, indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)

    s1 = sp.add_parser("single", help="单张图推理")
    s1.add_argument("--image", required=True)
    s1.add_argument("--ckpt", default=None)
    s1.add_argument("--top-k", type=int, default=5)
    s1.add_argument("--reject", type=float, default=0.0)
    s1.add_argument("--tta", default="flip",
                    choices=["none", "flip", "flip+scale", "5crop", "5crop+flip"])
    s1.add_argument("--temperature", type=float, default=1.0)
    s1.set_defaults(func=cmd_single)

    s2 = sp.add_parser("bench-tta", help="在测试目录上对比 TTA 策略")
    s2.add_argument("--test-dir", required=True)
    s2.add_argument("--ckpt", default=None)
    s2.add_argument("--limit", type=int, default=20)
    s2.add_argument("--img-size", type=int, default=None)
    s2.set_defaults(func=cmd_bench_tta)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()