"""FastAPI Web 应用：图片上传 + 多轮对话 + 工具调用日志展示。

接口：
- GET  /                       渲染聊天首页
- POST /api/upload             上传图片，返回 file_id 和 path
- POST /api/chat               {text, session_id, file_id?} → {session_id, answer, tool_logs, image_path}
- GET  /api/history/{sid}      返回指定 session 的所有消息（仅前端 debug）

启动：
    python -m src.web.app
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ..agent import agent_run, new_session
from ..config import get_web_host, get_web_port

WEB_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = WEB_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR = WEB_DIR / "templates"

app = FastAPI(title="看图识车 · 智能百科助手")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# 内存会话表（实训用，不需要持久化）
SESSIONS: Dict[str, List[dict]] = {}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)) -> JSONResponse:
    """上传图片后返回 file_id 与绝对路径（vision_tool 需要的是绝对路径）。"""
    suffix = Path(file.filename or "").suffix.lower() or ".jpg"
    if suffix not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        raise HTTPException(400, "只支持图片格式：jpg/png/bmp/webp")
    fid = f"{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}{suffix}"
    target = UPLOAD_DIR / fid
    content = await file.read()
    target.write_bytes(content)
    return JSONResponse({"file_id": fid, "path": str(target.resolve())})


@app.post("/api/chat")
async def chat(
    text: str = Form(...),
    session_id: Optional[str] = Form(None),
    file_id: Optional[str] = Form(None),
) -> JSONResponse:
    """跑一个 turn。file_id 可选，若给则让 vision 工具识别它。"""
    sid = session_id or uuid.uuid4().hex
    if sid not in SESSIONS:
        SESSIONS[sid] = new_session()

    user_text = text
    # 如果带了图，把图路径作为额外的 hint 告诉 agent（视觉工具按需调用）
    image_path: Optional[str] = None
    if file_id:
        p = UPLOAD_DIR / file_id
        if not p.is_file():
            raise HTTPException(400, f"upload 不存在: {file_id}")
        image_path = str(p.resolve())
        user_text = (
            f"{text}\n\n[系统提示] 这一轮用户上传了一张本地图片：{image_path}。"
            f"如果需要识别，请使用 classify_car 工具，并把 image_path 设为这个绝对路径。"
        )

    try:
        res = agent_run(SESSIONS[sid], user_text)
    except RuntimeError as e:
        return JSONResponse(
            {"session_id": sid, "answer": f"❌ 调用 DeepSeek 失败：{e}", "tool_logs": []},
            status_code=500,
        )
    except Exception as e:
        return JSONResponse(
            {"session_id": sid, "answer": f"❌ 服务器内部错误：{e}", "tool_logs": []},
            status_code=500,
        )

    return JSONResponse({
        "session_id": sid,
        "answer": res["answer"],
        "tool_logs": res["tool_logs"],
        "image_path": image_path,
    })


@app.get("/api/history/{sid}")
async def history(sid: str) -> JSONResponse:
    msgs = SESSIONS.get(sid)
    if msgs is None:
        return JSONResponse({"session_id": sid, "messages": []})
    return JSONResponse({"session_id": sid, "messages": msgs})


def run() -> None:
    import uvicorn
    uvicorn.run(
        "src.web.app:app",
        host=get_web_host(),
        port=get_web_port(),
        reload=False,
    )


if __name__ == "__main__":
    run()
