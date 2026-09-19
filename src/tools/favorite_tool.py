"""浏览/收藏历史工具（SQLite 实现）。

工具：
  - add_to_history(image_id, car_id, confidence, source)
  - list_history(limit, only_favorite)
  - mark_favorite(history_id, favorite=True)

注意：
  - car_id 可以为 "unknown"（不在 50 类里），照样记录。
  - 数据库文件写到 data_local/，不提交到仓库。
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict

from jsonschema import Draft202012Validator

from ..config import get_db_path

DB_PATH = get_db_path()


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                image_id TEXT,
                car_id TEXT,
                confidence REAL,
                source TEXT,
                favorite INTEGER DEFAULT 0
            )
            """
        )
        conn.commit()


init_db()


# ---------- 业务函数 ----------

def add_to_history(image_id: str, car_id: str, confidence: float, source: str = "user_upload") -> Dict[str, Any]:
    if not image_id:
        return {"ok": False, "error": "image_id 不能为空"}
    if car_id not in ("unknown",) and (len(car_id) != 4 or not car_id.isdigit()):
        return {"ok": False, "error": "car_id 必须是 4 位数字或字符串 unknown"}
    if not (0.0 <= float(confidence) <= 1.0):
        return {"ok": False, "error": "confidence 必须在 [0,1]"}

    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO history(image_id, car_id, confidence, source) VALUES(?,?,?,?)",
            (image_id, car_id, float(confidence), source),
        )
        conn.commit()
        return {"ok": True, "history_id": cur.lastrowid}


def list_history(limit: int = 20, only_favorite: bool = False) -> Dict[str, Any]:
    limit = max(1, min(int(limit), 200))
    with _connect() as conn:
        q = "SELECT id, created_at, image_id, car_id, confidence, source, favorite FROM history"
        if only_favorite:
            q += " WHERE favorite=1"
        q += " ORDER BY id DESC LIMIT ?"
        rows = conn.execute(q, (limit,)).fetchall()
    return {"ok": True, "items": [dict(r) for r in rows]}


def mark_favorite(history_id: int, favorite: bool = True) -> Dict[str, Any]:
    with _connect() as conn:
        cur = conn.execute("UPDATE history SET favorite=? WHERE id=?", (1 if favorite else 0, int(history_id)))
        conn.commit()
        if cur.rowcount == 0:
            return {"ok": False, "error": f"找不到 history_id={history_id}"}
        return {"ok": True, "history_id": int(history_id), "favorite": favorite}


# ---------- 工具定义 ----------

ADD_TO_HISTORY_SCHEMA = {
    "type": "object",
    "properties": {
        "image_id": {"type": "string", "description": "上传图片名，比如 001.jpg"},
        "car_id": {"type": "string", "description": "classes.txt 中的 4 位 ID 或 unknown"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "source": {"type": "string", "description": "来源标记，默认 user_upload"},
    },
    "required": ["image_id", "car_id", "confidence"],
    "additionalProperties": False,
}

LIST_HISTORY_SCHEMA = {
    "type": "object",
    "properties": {
        "limit": {"type": "integer", "minimum": 1, "maximum": 200, "default": 20},
        "only_favorite": {"type": "boolean", "default": False},
    },
    "additionalProperties": False,
}

MARK_FAVORITE_SCHEMA = {
    "type": "object",
    "properties": {
        "history_id": {"type": "integer"},
        "favorite": {"type": "boolean", "default": True},
    },
    "required": ["history_id"],
    "additionalProperties": False,
}


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "add_to_history",
            "description": "把刚刚识别的车型加入浏览历史，含 image_id/car_id/confidence。car_id 可以是 4 位数字或 unknown。",
            "parameters": ADD_TO_HISTORY_SCHEMA,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_history",
            "description": "查看最近的浏览/收藏记录，默认按时间倒序返回最多 limit 条。",
            "parameters": LIST_HISTORY_SCHEMA,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mark_favorite",
            "description": "把某条历史记录标记为收藏或取消收藏。",
            "parameters": MARK_FAVORITE_SCHEMA,
        },
    },
]


TOOL_FUNCTIONS = {
    "add_to_history": add_to_history,
    "list_history": list_history,
    "mark_favorite": mark_favorite,
}


# ---------- Agent 调用入口 ----------

def run(tool_call: dict, name: str) -> Dict[str, Any]:
    if name not in TOOL_FUNCTIONS:
        return {"ok": False, "error": f"工具不在白名单: {name}"}
    args_raw = tool_call.get("function", {}).get("arguments", "")
    try:
        args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
    except json.JSONDecodeError:
        return {"ok": False, "error": "参数不是合法的 JSON"}
    schema_map = {
        "add_to_history": ADD_TO_HISTORY_SCHEMA,
        "list_history": LIST_HISTORY_SCHEMA,
        "mark_favorite": MARK_FAVORITE_SCHEMA,
    }
    try:
        Draft202012Validator(schema_map[name]).validate(args)
    except Exception as e:
        return {"ok": False, "error": f"参数校验失败: {e}"}
    try:
        return TOOL_FUNCTIONS[name](**args)
    except Exception as e:
        return {"ok": False, "error": f"工具执行失败: {e}"}
