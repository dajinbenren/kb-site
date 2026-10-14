"""
llm_client.py —— 第 2 层底座：大模型客户端
=========================================
两个内核（RAG / Agent）都要调大模型，所以客户端必须共用一份，不能各写各的。

三个关键点（面试会问）：
1. 用 OpenAI SDK 但把 base_url 指向千问兼容模式 —— 换供应商只需改 base_url + model，
   业务代码一行不动。这叫「供应商可替换性」。
2. 对话模型和向量模型是「两个模型」，但必须来自同一家供应商。
   向量维度不匹配会导致检索全废。
3. 向量化要支持批量：一次传一个 list，让服务端的批处理生效，比循环单条快一个量级。
"""
import os
from dotenv import load_dotenv
from openai import OpenAI

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not API_KEY:
    raise SystemExit(
        "缺少 DASHSCOPE_API_KEY。请在项目根目录的 .env 里写：\n"
        "  DASHSCOPE_API_KEY=sk-你的千问key"
    )

BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen-plus")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-v3")
RERANK_MODEL = os.getenv("RERANK_MODEL", "gte-rerank-v2")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def chat(messages: list[dict], tools: list[dict] | None = None, temperature: float = 0.2) -> dict:
    """对话补全。传 tools 就进入 function calling 模式，不传就是普通问答。

    返回原始 message 对象：里面可能带 .content，也可能带 .tool_calls。
    「可能带」这三个字就是agent loop 存在的原因——模型这轮到底想不想调工具。
    """
    kwargs = dict(model=CHAT_MODEL, messages=messages, temperature=temperature)
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.model_dump(exclude_none=False)


EMBED_BATCH = int(os.getenv("EMBED_BATCH", "10"))  # 千问 embedding 接口单次最多 10 条


def embed(texts: list[str]) -> list[list[float]]:
    """批量向量化。

    两个坑（真实踩过）：
    1. 必须list 进 list 出，且顺序严格对应——后面 Chroma 靠这个顺序对齐原文。
    2. 服务端对单批条数有上限（千问 text-embedding-v3 最多 10 条），
       直接把 14 条丢过去会报 400 batch size is invalid。
       所以这里自动分批再拼回来，业务侧感觉不到。
    """
    if len(texts) <= EMBED_BATCH:
        resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
        return [d.embedding for d in resp.data]

    out = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i:i + EMBED_BATCH]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        out.extend(d.embedding for d in resp.data)
    return out


def rerank(query: str, docs: list[str], top_n: int = 3) -> list[int]:
    """Rerank 精排：用「重排序模型」对候选集重新打分，返回下标列表。

    为什么需要它（这是本项目最值钱的一个知识点）：
        向量检索是「粗排」，追求快，用双塔模型把 query 和 doc 各自编码成向量，
        算个余弦就完事——query 和 doc 之间没有任何交互，
        所以「紧急发布审批」和「发布窗口」这种词面相近但语义不同的块会混进来。
        Rerank 是「精排」，让 query 和每个 doc 拼在一起送进交叉编码模型，
        做逐对的深度交互，精度高但慢，只对 top-30 这种小候选集做。

        标准链路：BM25 + 向量 粗排召回 30 条 → Rerank 精排取 top-3 → 送 LLM。
        只用粗排的话，TOP_K 稍微一卡就像本项目「紧急发布审批」那道题，
        真答案排在第 4 名被挡在门外。
    """
    if not docs:
        return []

    # 坑：OpenAI SDK 没有 rerank 方法（实测 AttributeError），必须自己发 HTTP。
    # 而且千问两个排序模型的接口路径完全不同，且都不是 /compatible-mode/v1：
    #   qwen3-rerank     → POST /compatible-api/v1/reranks          （扁平结构）
    #   gte-rerank-v2    → POST /api/v1/services/rerank/text-rerank/text-rerank（input/parameters 嵌套）
    # 这里两条都试，任一成功即返回。
    import httpx

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    n = min(top_n, len(docs))
    attempts = [
        (
            "https://dashscope.aliyuncs.com/compatible-api/v1/reranks",
            {"model": "qwen3-rerank", "query": query, "documents": docs, "top_n": n,
             "return_documents": False},
            lambda j: [x["index"] for x in j["output"]["results"]],
        ),
        (
            "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
            {"model": RERANK_MODEL, "input": {"query": query, "documents": docs},
             "parameters": {"return_documents": False, "top_n": n}},
            lambda j: [x["index"] for x in j["output"]["results"]],
        ),
    ]

    for url, payload, pick in attempts:
        try:
            r = httpx.post(url, headers=headers, json=payload, timeout=30.0)
            r.raise_for_status()
            idx = pick(r.json())
            if idx:
                print(f"[rerank] 精排生效 via {url.rsplit('/', 1)[-1]}")
                return idx
        except Exception as e:
            print(f"[rerank] {url.rsplit('/', 1)[-1]} 失败（{type(e).__name__}）")

    # 降级：保持原顺序。精排是「锦上添花」，绝不能做单点故障。
    print("[rerank] 全部失败，降级为粗排原顺序")
    return list(range(n))


if __name__ == "__main__":
    # 自检：确认网络通、两个模型都可用
    msg = chat([{"role": "user", "content": "用一句话回答：1+1等于几？"}])
    print("[chat] ", CHAT_MODEL, "→", msg["content"].strip()[:60])
    vec = embed(["测试文本"])
    print("[embed]", EMBED_MODEL, "→ 向量维度", len(vec[0]))