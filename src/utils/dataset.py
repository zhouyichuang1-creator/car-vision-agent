"""数据集与 DataLoader。

约定目录布局（与教师下发一致）：
  数据集/train/<id>/xxx.jpg
  数据集/val/<id>/xxx.jpg
  其中 <id> 是 4 位字符串，如 "0042"。

我们会在外层加 `unknown/`。
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(folder: Path) -> List[Path]:
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])


def load_class_index(class_ids: Sequence[str]) -> Dict[str, int]:
    return {cid: i for i, cid in enumerate(class_ids)}


def list_class_folders(root: Path, exclude: Optional[Sequence[str]] = None) -> List[str]:
    """按 ID 升序返回所有类别目录名（不含 exclude）。"""
    exclude_set = set(exclude or [])
    out = []
    if not root.is_dir():
        return out
    for p in sorted(root.iterdir()):
        if p.is_dir() and p.name not in exclude_set:
            out.append(p.name)
    return out


class CarDataset(Dataset):
    """从按 ID 分子目录的目录读取。

    label_map: id -> 索引（0..N-1）。unknown 类可放最末尾 (N)。
    """

    def __init__(
        self,
        root: Path,
        label_map: Dict[str, int],
        transform: Optional[Callable] = None,
        limit_per_class: Optional[int] = None,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.label_map = dict(label_map)
        self.samples: List[Tuple[Path, int]] = []
        for cid in sorted(self.label_map.keys()):
            cdir = self.root / cid
            if not cdir.is_dir():
                continue
            imgs = list_images(cdir)
            if limit_per_class is not None:
                imgs = imgs[:limit_per_class]
            self.samples.extend((p, self.label_map[cid]) for p in imgs)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        # 第三项回传文件路径，便于 debug / 出错时定位
        return img, label, str(path)
