"""DeepSeek Agent 循环 + 多轮 Session。

核心思路：
- tools/ 目录下每个工具文件都暴露 (TOOL_DEFINITIONS, TOOL_FUNCTIONS, run)
- 本文件组装所有工具、注册到 agent，并暴露：
    - new_session() -> List[dict]
    - agent_run(session, user_text) -> {answer, tool_logs, messages}
    - 会话"图片 + 识别结果"上下文通过 SYSTEM + 上一轮 assistant 消息在 messages 里自然传递

设计原则（任务书 5.4、5.5）：
- 多轮：消息历史自然保留，识别结果在上一轮 assistant 消息里，新一轮能直接引用
- 低置信度：交给 SYSTEM / 用户问句明确，模型不该编造；这一步在 vision_tool 已经
  把 top_k 一并返回，让模型自然能给出 Top-K / 不确定时的澄清
"""
from __future__ import annotations

import copy
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests

from .config import get_api_key, get_model_name
from .tools import vision_tool, car_info_tool, favorite_tool


API_URL = "https://api.deepseek.com/chat/completions"

SYSTEM_PROMPT = """你是「看图识车 · 智能百科助手」，一位专业且用词通俗的车型顾问。

# 你掌握的工具
1. **classify_car(image_path)** —— 对单张图跑本组自训练的 51 类视觉模型，返回 top1_id（含 unknown）、top1_conf、top_k。
2. **query_car_info(car_id)** —— 查 classes.txt 中 50 类已注册车型的基础资料（中文名/英文名/类型）。只识别这 50 类内 ID。
3. **add_to_history(image_id, car_id, confidence, source="user")** —— 把这次识别结果存入浏览记录。
4. **list_history(limit=10, only_favorite=False)** —— 查询历史或收藏。
5. **mark_favorite(history_id, favorite=True)** —— 标记 / 取消收藏。

# 工作守则（必须严格遵守）
1. 看到任何车辆图片，**必须**先调 `classify_car` 拿到模型输出，再继续组织语言。禁止凭印象编造车型。
2. 模型返回 `top1_id == "unknown"`（字符串）时：**直接告诉用户「未能识别为本组支持的车型」**，不要硬猜、不要写进历史、不要 query_car_info。
3. confidence < 0.5 时，**必须**主动给 Top-3 候选 + 引导用户重拍或换角度。
4. 用户没说"记录/收藏"时，**不要**自动调 add_to_history；说"看看我看过什么"时才调 list_history。
5. 工具返回 `ok=false` 时，必须把 error 翻译成一句人话告诉用户，而不是原样回显。

# 输出格式（用 Markdown，前端会渲染）
- **首行**：用一行 emoji 简明开场（🔍 已识别 / ❓ 未能识别 / 📚 已查询 / ⭐ 已加入收藏 等）。
- **核心结论**：用加粗给出车型中文名 + 4 位 ID + 置信度百分比。
- **车型资料**（仅当 query_car_info 成功后）：用 3–5 个 bullet 列：英文名（name_en）、车身类型（body_type / body_type_detail）、品牌（brand_zh）、特征亮点。
- **Top-K 候选**（仅当 confidence < 0.5）：以表格或 bullet 给出 top3 的 ID + 置信度。
- **建议动作**：根据当前上下文给一句话的下一步（"可以让我加入收藏"、"试试换个角度重拍"等）。
- **语气**：专业但不冷漠，可以用「这款车」「它的外观」等自然表达；避免列点罗列，每个 bullet 配 8–20 字解释。
- **长度**：3–8 行，不要写小作文；不要重复工具调用结果原字段。
"""


# ---------- 工具注册 ----------

ALL_TOOL_DEFS: List[dict] = []
ALL_TOOL_FNS: Dict[str, Any] = {}


def _register(tool_mod) -> None:
    for td in tool_mod.TOOL_DEFINITIONS:
        ALL_TOOL_DEFS.append(td)
        name = td["function"]["name"]
        ALL_TOOL_FNS[name] = tool_mod.run


for _m in (vision_tool, car_info_tool, favorite_tool):
    _register(_m)


# ---------- API 调用 ----------

def request_completion(messages: List[dict], tools: Optional[List[dict]] = None) -> dict:
    api_key = get_api_key()
    payload = {
        "model": get_model_name(),
        "messages": messages,
        "thinking": {"type": "disabled"},
        "temperature": 0.2,
        "max_tokens": 2048,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    try:
        resp = requests.post(
            API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=(10, 90),
        )
    except requests.Timeout as e:
        raise RuntimeError("DeepSeek 请求超时") from e
    except requests.RequestException as e:
        raise RuntimeError(f"DeepSeek 网络异常: {e}") from e

    if not resp.ok:
        hints = {
            400: "检查消息顺序/工具定义",
            401: "API Key 无效或已过期",
            402: "DeepSeek 账号余额不足",
            403: "无权限访问该模型",
            429: "请求频率过高",
        }
        raise RuntimeError(f"DeepSeek HTTP {resp.status_code}: {hints.get(resp.status_code, '查平台')}")
    data = resp.json()
    if "choices" not in data:
        raise RuntimeError(f"DeepSeek 返回结构异常: {data}")
    return data


def assistant_message(api_response: dict) -> dict:
    return api_response["choices"][0]["message"]


def tool_result_message(tool_call: dict, result: dict) -> dict:
    return {
        "role": "tool",
        "tool_call_id": tool_call.get("id", ""),
        "content": json.dumps(result, ensure_ascii=False),
    }


def execute_tool(tool_call: dict, logs: List[dict]) -> dict:
    name = tool_call.get("function", {}).get("name", "")
    runner = ALL_TOOL_FNS.get(name)
    started = time.perf_counter()
    if runner is None:
        result = {"ok": False, "error": f"工具不在白名单: {name}"}
    else:
        try:
            result = runner(tool_call, name)
        except Exception as e:
            result = {"ok": False, "error": f"工具执行异常: {type(e).__name__}: {e}"}
    logs.append({
        "time": datetime.now(timezone.utc).isoformat(),
        "tool_call_id": tool_call.get("id"),
        "name": name,
        "arguments_raw": tool_call.get("function", {}).get("arguments", ""),
        "result": result,
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
    })
    return result


# ---------- 会话 + 主循环 ----------

@dataclass
class Turn:
    user_text: str
    assistant_text: str = ""
    tool_logs: List[dict] = field(default_factory=list)


def new_session(system_prompt: Optional[str] = None) -> List[dict]:
    return [{"role": "system", "content": system_prompt or SYSTEM_PROMPT}]


def agent_run(
    session: List[dict],
    user_text: str,
    tools: Optional[List[dict]] = None,
    max_rounds: int = 8,
    max_tool_calls: int = 12,
) -> dict:
    """在给定 session 上跑一个 turn，返回 {answer, tool_logs}。"""
    allowed_tools = tools if tools is not None else ALL_TOOL_DEFS
    allowed_names = {t["function"]["name"] for t in allowed_tools}

    working = copy.deepcopy(session)
    working.append({"role": "user", "content": user_text})

    tool_logs: List[dict] = []
    call_count = 0
    final_answer = ""

    for round_no in range(1, max_rounds + 1):
        api_resp = request_completion(working, tools=allowed_tools)
        msg = assistant_message(api_resp)
        tool_calls = msg.get("tool_calls") or []

        if not tool_calls:
            final_answer = msg.get("content", "") or ""
            working.append(msg)
            break

        working.append(msg)

        for tc in tool_calls:
            if call_count >= max_tool_calls:
                working.append(tool_result_message(tc, {"ok": False, "error": "tool_calls 超过上限"}))
                continue
            call_count += 1
            name = tc.get("function", {}).get("name", "")
            if name not in allowed_names:
                result = {"ok": False, "error": f"工具不在本轮允许列表: {name}"}
                tool_logs.append({
                    "time": datetime.now(timezone.utc).isoformat(),
                    "tool_call_id": tc.get("id"),
                    "name": name,
                    "arguments_raw": tc.get("function", {}).get("arguments", ""),
                    "result": result,
                    "elapsed_ms": 0,
                })
            else:
                result = execute_tool(tc, tool_logs)
            working.append(tool_result_message(tc, result))

    if not final_answer:
        api_resp = request_completion(working, tools=allowed_tools)
        msg = assistant_message(api_resp)
        final_answer = msg.get("content", "") or ""
        working.append(msg)

    session[:] = working
    return {"answer": final_answer, "tool_logs": tool_logs, "messages": list(working)}


# ---------- CLI 调试入口 ----------

def cmd_smoke(_: argparse.Namespace) -> None:
    sess = new_session()
    res = agent_run(
        sess,
        "你好，做个自我介绍，你都能做什么？",
        max_rounds=2,
        max_tool_calls=3,
    )
    print("=== 回答 ===")
    print(res["answer"])
    print("=== 工具调用 ===")
    for log in res["tool_logs"]:
        print(json.dumps(log, ensure_ascii=False, indent=2))


import argparse  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("smoke", help="不调视觉工具，仅验证 DeepSeek API + agent 循环")
    s.set_defaults(func=cmd_smoke)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()