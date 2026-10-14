"""
main.py —— 第 3 层：FastAPI 服务层
=================================
把两个内核包成 HTTP 接口 + 静态前端。这层是「壳」，不产生智能。

三个 FastAPI 卖点（面试必答）：
1. 自带 /docs 交互式API 文档：把Pydantic 模型当 OpenAPI schema 自动生成，
   不用手写接口文档，前端同学照着调。
2. 原生 async：调大模型是 IO 密集（等待网络），
   同步 def 会占住线程池，改成 async def 一个进程能并发处理多个请求。
   注意：我们的 llm_client 是同步 SDK，所以接口层用普通 def，
   FastAPI 会自动丢到线程池，反而是对的。这点面试常追问。
3. Pydantic 做入参校验：question 传成数字或空字符串直接 422，
   不用等进到内核才发现问题。

会话存储用 SQLite（Python 自带，零安装）：
    会话表存 title/mode/时间，对话表存每轮问答。
    为什么不用内存 list？因为服务重启就没了；
    为什么不上Postgres？本地小项目够用，且 SQLite 免安装是优势。
"""
import os
import sqlite3
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import vector_store
import kernel_rag
import kernel_agent

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(BASE_DIR, "app.db")


# ============================================================
# 数据层：SQLite 建表 + 读写
# ============================================================
def db_conn():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with db_conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id   TEXT PRIMARY KEY,
                title TEXT,
                mode  TEXT,
                created_at REAL
            )""")
        c.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role       TEXT,
                content    TEXT,
                payload    TEXT,
                created_at REAL,
                FOREIGN KEY(session_id) REFERENCES sessions(id)
            )""")
    print("[db] 会话表就绪")


def save_session(session_id: str, title: str, mode: str):
    with db_conn() as c:
        c.execute(
            "INSERT OR REPLACE INTO sessions(id,title,mode,created_at) VALUES(?,?,?,?)",
            (session_id, title, mode, time.time()),
        )


def save_message(session_id: str, role: str, content: str, payload: dict | None = None):
    import json

    with db_conn() as c:
        c.execute(
            "INSERT INTO messages(session_id,role,content,payload,created_at) VALUES(?,?,?,?,?)",
            (session_id, role, content, json.dumps(payload or {}, ensure_ascii=False), time.time()),
        )


def load_history(session_id: str, limit: int = 6) -> list[dict]:
    """取最近几轮，转成模型能吃的 messages 格式。"""
    with db_conn() as c:
        rows = c.execute(
            "SELECT role, content FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


def list_sessions() -> list[dict]:
    with db_conn() as c:
        rows = c.execute(
            "SELECT s.*, (SELECT COUNT(*) FROM messages m WHERE m.session_id=s.id) AS n "
            "FROM sessions s ORDER BY created_at DESC LIMIT 30"
        ).fetchall()
    return [dict(r) for r in rows]


# ============================================================
# 生命周期：启动时建库
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    try:
        n = vector_store.build()
        s = vector_store.stats()
        print(f"[启动] 向量库就绪：{n} 块")
        if s.get("degenerate"):
            print(f"[警告] 库总块数 <= TOP_K，检索会退化，建议加大语料或调小 CHUNK_SIZE")
    except Exception as e:
        print(f"[启动] 向量库初始化失败（服务仍可启动）：{e}")
    yield


app = FastAPI(title="星辰知识库助手", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")


# ============================================================
# 请求/响应模型
# ============================================================
class AskReq(BaseModel):
    question: str = Field(..., min_length=1, max_length=500, description="用户问题")
    session_id: str | None = None
    mode: Literal["rag", "agent"] = "rag"
    top_k: int | None = Field(None, ge=1, le=10)


@app.get("/")
def index():
    return FileResponse(os.path.join(BASE_DIR, "static", "index.html"))


@app.get("/api/health")
def health():
    s = vector_store.stats()
    return {
        "status": "ok",
        "chat_model": __import__("llm_client").CHAT_MODEL,
        "embed_model": __import__("llm_client").EMBED_MODEL,
        "vector_db": s,
    }


@app.post("/api/ask")
def ask(req: AskReq):
    q = req.question.strip()
    sid = req.session_id or uuid.uuid4().hex[:12]

    history = load_history(sid) if req.session_id else None

    try:
        if req.mode == "rag":
            result = kernel_rag.answer(q, top_k=req.top_k, use_context=history)
        else:
            result = kernel_agent.run(q, use_context=history)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"内核执行失败：{e}")

    save_session(sid, q[:20], req.mode)
    save_message(sid, "user", q)
    save_message(sid, "assistant", result["answer"],
                 {"mode": result["mode"], "sources": result.get("sources", []),
                  "steps": result.get("steps", []), "turns": result.get("turns")})

    return {"session_id": sid, **result}


@app.get("/api/sessions")
def sessions():
    return list_sessions()


@app.get("/api/sessions/{sid}")
def session_detail(sid: str):
    with db_conn() as c:
        rows = c.execute(
            "SELECT role,content,payload,created_at FROM messages WHERE session_id=? ORDER BY id",
            (sid,),
        ).fetchall()
    import json

    out = []
    for r in rows:
        p = json.loads(r["payload"] or "{}")
        out.append({"role": r["role"], "content": r["content"], "time": r["created_at"],
                    "sources": p.get("sources", []), "steps": p.get("steps", []),
                    "turns": p.get("turns")})
    return {"session_id": sid, "messages": out}


@app.post("/api/rebuild")
def rebuild():
    n = vector_store.build(force=True)
    return {"ok": True, "chunks": n}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)