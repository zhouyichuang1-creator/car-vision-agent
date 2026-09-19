"""视觉分类工具 —— 把训练好的模型包装成 DeepSeek Tool。

输入：image_path（绝对路径）。
输出：{top1_id, top1_conf, top_k, predicted_label, note?}
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from jsonschema import Draft202012Validator

from ..config import get_model_ckpt
from ..inference import predict

TOOL_NAME = "classify_car"

TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "image_path": {
            "type": "string",
            "description": "图片的绝对路径，比如 /tmp/upload/abc.jpg",
        }
    },
    "required": ["image_path"],
    "additionalProperties": False,
}

TOOL_DEFINITIONS = [{
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": (
            "使用本组自训练的 51 类图像分类模型对单张图片做识别"
            "（推理启用水平翻转 TTA，并按验证集标定的阈值做 unknown 拒识回退）。"
            "返回 top1_id（classes.txt 中的 4 位 ID，或字符串 unknown）、"
            "top1_conf（0~1 的置信度）以及 top_k 候选。"
            "当模型预测为 unknown 时不要编造具体车型；"
            "当 top1_conf 低于 0.5 时不要武断回答，可提示用户换角度重拍。"
        ),
        "parameters": TOOL_SCHEMA,
    },
}]


def classify_car(image_path: str, top_k: int = 5) -> Dict[str, Any]:
    """业务侧执行函数。返回结构化结果，不抛异常（异常也包到 dict 里）。"""
    p = Path(image_path)
    if not p.is_file():
        return {"ok": False, "error": f"找不到图片: {image_path}"}
    try:
        # flip TTA（5-crop 在我们数据上反而掉点）+ 验证集标定的 unknown 拒识回退
        res = predict(image_path=p, ckpt_path=get_model_ckpt(),
                      top_k=top_k, tta="flip", temperature=1.0)
    except Exception as e:
        return {"ok": False, "error": f"模型推理失败: {e}"}

    return {
        "ok": True,
        "image_id": res["image_id"],
        "top1_id": res["top1_id"],
        "top1_conf": round(res["top1_conf"], 6),
        "top_k": [{"id": i, "confidence": round(c, 6)} for i, c in res["top_k"]],
        "note": res.get("note"),
        "img_size_used": res.get("img_size_used"),
        "tta": res.get("tta"),
        "reject_tau": res.get("reject_tau"),
    }


TOOL_FUNCTIONS = {"classify_car": classify_car}


def run(tool_call: dict, name: str | None = None) -> Dict[str, Any]:
    """agent.py 调用的统一入口：先做参数校验，再执行具体函数。"""
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