"""车型信息工具 —— 根据 car_id 查 class_info.json，返回名称与粗略类型。

对应任务书"其他工具"里的一个。
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from jsonschema import Draft202012Validator

from ..config import CLASSES_FILE, CLASS_INFO_JSON

TOOL_NAME = "query_car_info"

TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "car_id": {
            "type": "string",
            "pattern": r"^\d{4}$",
            "description": "classes.txt 中的 4 位车型 ID",
        }
    },
    "required": ["car_id"],
    "additionalProperties": False,
}

TOOL_DEFINITIONS = [{
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": (
            "查询某车型的基础资料：中文名、英文名、车身类型分类。输入 car_id 是 classes.txt 中"
            "4 位带前导零的车型 ID，且必须属于本组已注册的 50 类；不属于时返回失败。"
        ),
        "parameters": TOOL_SCHEMA,
    },
}]


def _load() -> List[Dict]:
    if not CLASS_INFO_JSON.exists():
        return []
    with open(CLASS_INFO_JSON, encoding="utf-8") as f:
        return json.load(f)


def _selected_ids() -> set:
    """本组已注册的 50 个车型 ID（classes.txt）。"""
    if not CLASSES_FILE.exists():
        return set()
    return {
        line.strip()
        for line in CLASSES_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def query_car_info(car_id: str) -> Dict[str, Any]:
    selected = _selected_ids()
    if selected and car_id not in selected:
        return {
            "ok": False,
            "error": (
                f"{car_id} 不属于本组已注册的 50 个车型，无法提供资料。"
                "如果图片识别结果为 unknown，请直接告知用户未识别为本组支持的车型。"
            ),
        }
    rows = _load()
    for row in rows:
        if row["id"] == car_id:
            return {
                "ok": True,
                "id": row["id"],
                "name_from_new": row.get("name_from_new"),
                "name_from_old": row.get("name_from_old"),
                "type_info": row.get("type_info"),
            }
    return {"ok": False, "error": f"未知 car_id: {car_id}（不是 50 类之一，或 ID 拼写错）"}


TOOL_FUNCTIONS = {"query_car_info": query_car_info}


def run(tool_call: dict, name: str | None = None) -> Dict[str, Any]:
    name = name or tool_call.get("function", {}).get("name", "")
    if name not in TOOL_FUNCTIONS:
        return {"ok": False, "error": f"工具不在白名单: {name}"}
    args_raw = tool_call.get("function", {}).get("arguments", "")
    try:
        args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
    except json.JSONDecodeError:
        return {"ok": False, "error": "参数不是合法的 JSON"}
    try:
        Draft202012Validator(TOOL_SCHEMA).validate(args)
    except Exception as e:
        return {"ok": False, "error": f"参数校验失败: {e}"}
    try:
        return TOOL_FUNCTIONS[name](**args)
    except Exception as e:
        return {"ok": False, "error": f"工具执行失败: {e}"}
