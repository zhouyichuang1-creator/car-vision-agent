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


def _parse_name(name_from_new: Any, name_from_old: Any) -> Dict[str, str]:
    """把 class_info.json 的名称字段拆成干净的中英文。

    原始格式：name_from_new = ``品牌_车型``；name_from_old = ``品牌_车型==Brand_Model``。
    """
    zh_full = str(name_from_new or "").strip()
    old = str(name_from_old or "").strip()
    zh_part, _, en_part = old.partition("==")
    en_part = en_part.strip() or zh_part.strip()

    brand_zh, _, model_zh = zh_full.partition("_")
    brand_en, _, model_en = en_part.partition("_")

    return {
        "brand_zh": brand_zh.strip() or zh_full,
        "model_zh": model_zh.strip() or zh_full,
        "name_zh": zh_full.replace("_", " ").strip(),
        "brand_en": brand_en.strip() or en_part,
        "model_en": model_en.strip() or en_part,
        "name_en": en_part.replace("_", " ").strip(),
    }


def _parse_type(type_info: Any) -> Dict[str, str]:
    """把 type_info 拆成车身类型的干净字段。

    原始格式：``大类中文####子类中文==大类英文####子类英文``，
    例如 ``SUV####中型SUV==SUV####Mid-size SUV``。
    """
    raw = str(type_info or "").strip()
    segs = [s.strip() for s in raw.split("####")] if raw else []
    body_type = segs[0] if segs else ""
    detail, group_en = "", ""
    if len(segs) >= 2:
        zh, _, en = segs[1].partition("==")
        detail = zh.strip()
        group_en = en.strip()
    detail_en = segs[2].strip() if len(segs) >= 3 else ""
    return {
        "body_type": body_type,
        "body_type_detail": detail,
        "body_type_en": detail_en,
        "body_type_group_en": group_en,
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
            out: Dict[str, Any] = {"ok": True, "id": row["id"]}
            out.update(_parse_name(row.get("name_from_new"), row.get("name_from_old")))
            out.update(_parse_type(row.get("type_info")))
            return out
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
