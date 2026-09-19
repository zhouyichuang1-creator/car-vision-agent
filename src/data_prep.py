"""数据准备：
  1. 读 class_info.json 统计每类的样本数 / 分布；
  2. 给出"50 类的候选方案"（按一定规则推荐，便于用户拍板）；
  3. 把"非 50 类"的图片采样合并成 unknown 训练集 / 验证集；
  4. 写 classes.txt（恰好 50 行，按 ID 升序，UTF-8，无 BOM）。

classes.txt 是任务书 4.1 规定的"唯一种类登记"，本脚本是其权威生成器。
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from .config import (
    CLASS_INFO_JSON,
    DATA_LOCAL,
    TRAIN_DIR,
    VAL_DIR,
)
from .utils.dataset import IMG_EXTS, list_images


UNKNOWN_DIR_NAME = "unknown"


# ---------------------- 工具函数 ----------------------

def load_class_info() -> List[dict]:
    with open(CLASS_INFO_JSON, encoding="utf-8") as f:
        return json.load(f)


def count_samples_per_class(folder: Path, ids: Sequence[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for cid in ids:
        d = folder / cid
        if not d.is_dir():
            out[cid] = 0
            continue
        out[cid] = sum(1 for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)
    return out


def write_classes_txt(selected_ids: Sequence[str], path: Path = Path("classes.txt")) -> None:
    sorted_ids = sorted({i for i in selected_ids if i})
    if len(sorted_ids) != 50:
        raise ValueError(f"必须恰好 50 个 ID，实际 {len(sorted_ids)} 个")
    bad = [i for i in sorted_ids if len(i) != 4 or not i.isdigit()]
    if bad:
        raise ValueError(f"非法 ID: {bad}")
    content = "\n".join(sorted_ids) + "\n"  # 文件以换行结尾
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)
    print(f"已写 {path}（{len(sorted_ids)} 行）")


# ---------------------- 选类候选 ----------------------

def suggest_candidates(
    classes: List[dict],
    counts: Dict[str, int],
    target: int = 50,
    seed: int = 42,
    min_count: int = 60,
) -> List[str]:
    """一个简单的启发式：样本数 ≥ min_count 的类里按 seed 采样 50 类。

    真实使用里你也可以手动改 classes.txt：
       先用 `python -m src.data_prep --make-classes 50` 生成初稿，
       再用文本编辑器调整，然后再跑下面 `--write` 一次落盘。
    """
    pool = [c["id"] for c in classes if counts.get(c["id"], 0) >= min_count]
    random.seed(seed)
    random.shuffle(pool)
    return sorted(pool[:target])


# ---------------------- unknown 采样 ----------------------

def make_unknown_from_others(
    train_root: Path,
    val_root: Path,
    selected_ids: Sequence[str],
    seed: int = 42,
    max_train: int = 2000,
    max_val: int = 600,
    per_source_max_train: int = 40,
    per_source_max_val: int = 15,
    class_info: List[dict] | None = None,
):
    """从所有"非 50 类"的图采样合并成 unknown/。

    - 总采样量有上限，防止 unknown 过大破坏训练比例。
    - 每个源类别的采样也有上限，避免 unknown 集中在某一种车型。
    - 训练、验证样本不重复。
    """
    selected = set(selected_ids)
    rng = random.Random(seed)

    if class_info is None:
        class_info = load_class_info()
    # 所有可采样的源类 = 全部类 - 已选 50 类
    src_ids = [c["id"] for c in class_info if c["id"] not in selected]
    print(f"  unknown 候选源类数: {len(src_ids)}")

    # 训练集采样
    train_out = train_root / UNKNOWN_DIR_NAME
    val_out = val_root / UNKNOWN_DIR_NAME
    train_out.mkdir(parents=True, exist_ok=True)
    val_out.mkdir(parents=True, exist_ok=True)
    # 清空旧的
    for p in train_out.iterdir():
        if p.is_file():
            p.unlink()
    for p in val_out.iterdir():
        if p.is_file():
            p.unlink()

    train_count = 0
    val_count = 0
    src_train_used: Dict[str, int] = {}
    src_val_used: Dict[str, int] = {}

    # 打散源类，对每个源类取前 per_source_max
    rng.shuffle(src_ids)
    for cid in src_ids:
        if train_count >= max_train and val_count >= max_val:
            break

        src_train_dir = train_root / cid
        if not src_train_dir.is_dir():
            continue
        train_files = list_images(src_train_dir)
        rng.shuffle(train_files)
        kept_for_train = train_files[:per_source_max_train]
        kept_for_train = kept_for_train[: max(0, max_train - train_count)]
        for p in kept_for_train:
            dst = train_out / f"{cid}_{p.name}"
            if dst.exists():
                continue
            shutil.copy2(p, dst)
            train_count += 1
            src_train_used[cid] = src_train_used.get(cid, 0) + 1

        src_val_dir = val_root / cid
        if not src_val_dir.is_dir():
            continue
        val_files = list_images(src_val_dir)
        rng.shuffle(val_files)
        kept_for_val = val_files[:per_source_max_val]
        kept_for_val = kept_for_val[: max(0, max_val - val_count)]
        for p in kept_for_val:
            dst = val_out / f"{cid}_{p.name}"
            if dst.exists():
                continue
            shutil.copy2(p, dst)
            val_count += 1
            src_val_used[cid] = src_val_used.get(cid, 0) + 1

    return {
        "train_files": train_count,
        "val_files": val_count,
        "src_classes_used": len(src_train_used),
        "train_per_src_avg": (train_count / max(1, len(src_train_used))),
    }


# ---------------------- CLI ----------------------

def cmd_inspect(_: argparse.Namespace) -> None:
    classes = load_class_info()
    train_counts = count_samples_per_class(TRAIN_DIR, [c["id"] for c in classes])
    val_counts = count_samples_per_class(VAL_DIR, [c["id"] for c in classes])
    total_train = sum(train_counts.values())
    total_val = sum(val_counts.values())
    print(f"类别数: {len(classes)}（id 范围 {classes[0]['id']} ~ {classes[-1]['id']}）")
    print(f"训练图总数: {total_train}, 验证图总数: {total_val}")
    print("训练样本数分布（前 5 / 后 5）:")
    items = sorted(train_counts.items(), key=lambda x: int(x[0]))
    for cid, n in items[:5]:
        print(f"  train/{cid}: {n}")
    print("  ...")
    for cid, n in items[-5:]:
        print(f"  train/{cid}: {n}")


def cmd_make_classes(args: argparse.Namespace) -> None:
    classes = load_class_info()
    train_counts = count_samples_per_class(TRAIN_DIR, [c["id"] for c in classes])
    cand = suggest_candidates(classes, train_counts, target=50, seed=args.seed, min_count=args.min_count)
    out = Path(args.out)
    write_classes_txt(cand, out)
    print(f"提示：可直接编辑 {out}（按 ID 升序、50 行、UTF-8），然后跑 --write-final 重新落盘。")


def cmd_make_unknown(args: argparse.Namespace) -> None:
    classes_path = Path(args.classes)
    if not classes_path.exists():
        raise SystemExit(f"找不到 {classes_path}，请先 --make-classes")
    selected_ids = [l.strip() for l in classes_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if len(selected_ids) != 50:
        raise SystemExit(f"classes.txt 必须是 50 行，实际 {len(selected_ids)}")
    classes = load_class_info()
    info = make_unknown_from_others(
        TRAIN_DIR,
        VAL_DIR,
        selected_ids,
        seed=args.seed,
        max_train=args.max_train,
        max_val=args.max_val,
        per_source_max_train=args.per_src_train,
        per_source_max_val=args.per_src_val,
        class_info=classes,
    )
    print("unknown 采样结果:", info)


def cmd_write_final(args: argparse.Namespace) -> None:
    classes_path = Path(args.classes)
    selected_ids = [l.strip() for l in classes_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    write_classes_txt(selected_ids, Path(args.out))


def main() -> None:
    p = argparse.ArgumentParser(description="数据准备：候选 50 类 + unknown 采样 + 写 classes.txt")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("inspect", help="统计类别与样本数分布")
    sp.set_defaults(func=cmd_inspect)

    sp = sub.add_parser("make-classes", help="生成 50 类候选 classes.txt 初稿（可手动改）")
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--min-count", type=int, default=60, help="每个被选类最少样本数（避免选过稀的类）")
    sp.add_argument("--out", default="classes.txt")
    sp.set_defaults(func=cmd_make_classes)

    sp = sub.add_parser("make-unknown", help="从未选类采样生成 unknown/ 数据")
    sp.add_argument("--classes", default="classes.txt")
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--max-train", type=int, default=2000)
    sp.add_argument("--max-val", type=int, default=600)
    sp.add_argument("--per-src-train", type=int, default=40)
    sp.add_argument("--per-src-val", type=int, default=15)
    sp.set_defaults(func=cmd_make_unknown)

    sp = sub.add_parser("write-final", help="把用户改过的 classes.txt 按规范重新落盘（ID 升序）")
    sp.add_argument("--classes", default="classes.txt")
    sp.add_argument("--out", default="classes.txt")
    sp.set_defaults(func=cmd_write_final)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
