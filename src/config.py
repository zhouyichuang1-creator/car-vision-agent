"""全局配置：所有敏感信息（API Key）只通过环境变量读，永远不写进文件。

加载顺序（均为"读"，从不"写"）：
  1. 进程环境变量 DEEPSEEK_API_KEY
  2. 本机 .env 文件（不会被 git 跟踪，详见 .gitignore）
"""
from __future__ import annotations

import os
from pathlib import Path

# 仅当本机存在 .env 时自动加载；project 仓库内不携带任何 Key
try:
    from dotenv import load_dotenv  # type: ignore

    _env_path = PROJECT_ROOT / ".env" if (PROJECT_ROOT := Path(__file__).resolve().parents[1]) else None
    if _env_path and _env_path.exists():
        load_dotenv(dotenv_path=_env_path, override=False)
except Exception:
    pass


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "数据集"
CLASS_INFO_JSON = DATA_ROOT / "cls_info" / "class_info.json"
TRAIN_DIR = DATA_ROOT / "train"
VAL_DIR = DATA_ROOT / "val"

CKPT_DIR = PROJECT_ROOT / "checkpoints"
DATA_LOCAL = PROJECT_ROOT / "data_local"
LOGS_DIR = PROJECT_ROOT / "logs"
CLASSES_FILE = PROJECT_ROOT / "classes.txt"


def get_api_key() -> str:
    """从环境变量读 DeepSeek API Key，绝不返回 None 之外的兜底字符串。"""
    key = (os.getenv("DEEPSEEK_API_KEY") or "").strip()
    if not key:
        raise RuntimeError(
            "未找到 DEEPSEEK_API_KEY 环境变量。请在 .env 或 PowerShell 里配置，"
            "不要把它硬编码到任何文件里。"
        )
    return key


def get_model_name() -> str:
    return (os.getenv("DEEPSEEK_MODEL") or "deepseek-v4-flash").strip()


# unknown 拒识回退阈值（在验证集 1928 张上标定，见 report 4.5 节）
# 规则：51 类 argmax 为 unknown 时，若"最佳已知类"概率 >= 此值则改判为该已知类
# v2 模型（30ep + MixUp + unknown 降采样）标定结论：tau >= 0.40 时收益完全持平，
# 说明模型自身的 unknown 判定已校准良好，不再需要事后回退；0.90 仅作极端兜底。
REJECT_TAU_DEFAULT = 0.90


def get_reject_threshold(default: float = REJECT_TAU_DEFAULT) -> float:
    """unknown 拒识回退阈值 tau（验证集标定值 0.35；<=0 表示关闭该规则）。"""
    try:
        return float(os.getenv("REJECT_THRESHOLD", default))
    except ValueError:
        return default


def get_top_k(default: int = 5) -> int:
    try:
        return int(os.getenv("TOP_K", default))
    except ValueError:
        return default


def get_web_host(default: str = "127.0.0.1") -> str:
    return (os.getenv("WEB_HOST") or default).strip()


def get_web_port(default: int = 8000) -> int:
    try:
        return int(os.getenv("WEB_PORT", default))
    except ValueError:
        return default


def get_model_ckpt() -> Path:
    p = os.getenv("MODEL_CKPT", "checkpoints/best.pt")
    path = PROJECT_ROOT / p
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_db_path() -> Path:
    p = os.getenv("DB_PATH", "data_local/favorites.db")
    path = PROJECT_ROOT / p
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
