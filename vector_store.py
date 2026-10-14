"""
vector_store.py —— 第 2 层底座：向量库封装
==========================================
把 Chroma 的细节关起来，对外只暴露三个动作：
    build()          建库（切分 + 向量化 + 写入）
    search(query,k)  检索 top-k
    stats()          看库的状态 ← 这个是最重要的调试函数

为什么要自己封一层？
1. 换向量库（Chroma / Milvus / pgvector）时只改这一个文件，RAG 内核一行不改。
2. stats() 能让你随时回答「我这套 RAG 到底有没有在检索」。上一轮实测：
   2511 字语料只切出 2 块，TOP_K=3 > 库中总数，检索退化成全量注入——
   那一刻如果在调RAG，几乎所有人都会得出「我的 RAG 效果很好」的错误结论。

切分逻辑（面试要能讲清）：
    先按段落聚合，段落超长再按字符窗口滑切。
    step = CHUNK_SIZE - CHUNK_OVERLAP
    相邻块共享 OVERLAP 个字符，防止一句完整的话被切成两半塞进两块。
"""
import os
import glob
import re

import chromadb
from rank_bm25 import BM25Okapi

import llm_client

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_DIR = os.path.join(BASE_DIR, "chroma_db")
# Chroma 1.5+ 要求集合名 3-512 字符，仅 [a-zA-Z0-9._-]，所以不能叫 "kb"
COLLECTION = "kb_docs"

# ---------- 三个关键参数 ----------
CHUNK_SIZE = 450      # 每块上限（字符）
CHUNK_OVERLAP = 80# 块间重叠，防边界断句
TOP_K = 4             # 检索返回块数 ← 不是越大越好


# ---------- 切分 ----------
def load_and_split() -> list[dict]:
    chunks = []
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.md")))
    if not files:
        raise SystemExit(f"data/ 下没有 .md 文件，请先放知识库文档。")

    for fpath in files:
        with open(fpath, encoding="utf-8") as f:
            text = f.read()
        fname = os.path.basename(fpath)

        # 按 Markdown 标题分节，标题一并保留在块里（否则「第八条」会孤立无上下文）
        sections, buf, title = [], "", ""
        for line in text.split("\n"):
            if line.startswith("#"):
                if buf.strip():
                    sections.append((title, buf))
                title, buf = line.strip(), ""
            else:
                buf += line + "\n"
        if buf.strip():
            sections.append((title, buf))

        # 过长的小节再按字符窗口滑切
        for sec_title, body in sections:
            if len(body) <= CHUNK_SIZE:
                chunks.append({"text": f"{sec_title}\n{body}".strip(), "source": fname})
                continue
            step = CHUNK_SIZE - CHUNK_OVERLAP
            for i in range(0, len(body), step):
                piece = body[i:i + CHUNK_SIZE].strip()
                if piece:
                    chunks.append({"text": f"{sec_title}\n{piece}".strip(), "source": fname})

    return chunks


# ---------- 建库 ----------
def build(force: bool = False):
    if os.path.exists(DB_DIR) and not force:
        try:
            n = stats()["total"]
            if n:
                print(f"[向量库] 已存在（{n} 块），跳过建库。加 --rebuild 可强制重建。")
                return n
        except Exception:
            pass

    chunks = load_and_split()
    print(f"[切分] {len(chunks)} 块来自 {len(set(c['source'] for c in chunks))} 个文件")
    vectors = llm_client.embed([c["text"] for c in chunks])

    chroma = chromadb.PersistentClient(path=DB_DIR)

    # 重建时用 Chroma 自己的 delete_collection 清空，
    # 而不是 shutil.rmtree 删目录——删目录在容器/只读环境下常被拦，
    # 而且会连带把整个库目录的元数据一起干掉。
    try:
        chroma.delete_collection(name=COLLECTION)
        print("[建库] 已清空旧集合")
    except Exception:
        pass  # 集合本来就不存在，正常

    col = chroma.create_collection(name=COLLECTION)
    col.add(
        ids=[f"c{i}" for i in range(len(chunks))],
        documents=[c["text"] for c in chunks],
        embeddings=vectors,
        metadatas=[{"source": c["source"], "idx": i} for i, c in enumerate(chunks)],
    )
    print(f"[建库] 完成：{len(chunks)} 块 → {DB_DIR}")
    return len(chunks)


# ---------- 检索 ----------
_TOKEN_RE = re.compile(r"[\w一-鿿]+")


def _tokenize(text: str) -> list[str]:
    """中文没有空格，按字bigram 切；英文数字按单词切。

    这是中文关键词检索的标准做法：把「紧急发布审批」切成
    紧急/急发/发布/布审/审批 五个 token，才能命中「紧急发布」和「发布审批」。
    """
    toks = []
    for seg in _TOKEN_RE.findall(text):
        if re.fullmatch(r"[一-鿿]+", seg) and len(seg) > 1:
            toks.extend([seg[i:i + 2] for i in range(len(seg) - 1)])
        else:
            toks.append(seg.lower())
    return toks


def search(query: str, top_k: int = TOP_K, mode: str = "hybrid", rerank: bool = True) -> list[dict]:
    """检索。

    mode:
      "vector"  只用向量语义相似度
      "keyword" 只用 BM25 字面匹配
      "hybrid"  RRF 融合（默认）

    为什么需要混合检索（真实踩坑）：
        纯向量下「紧急发布审批 / 紧急发布流程 / 例外发布」三个不同关键词
        返回完全同一批块——换词不改变向量空间的邻域关系。
        BM25 看字面词频，能精确命中但同义改写就废。
        所以用 RRF（Reciprocal Rank Fusion）按「排名」而非「分数」融合，
        天然规避了两边分数量纲不可比的问题。

    rerank:
        粗排（混合检索）召回更宽的候选，再用重排序模型精排。
        对应生产链路：粗排 30 → 精排 top_k → 送 LLM。
    """
    chroma = chromadb.PersistentClient(path=DB_DIR)
    col = chroma.get_or_create_collection(name=COLLECTION)
    total = col.count()
    all_docs = col.get(include=["documents", "metadatas"])

    # ---- 1. 向量召回 ----
    q_vec = llm_client.embed([query])[0]
    # 粗排阶段多取一些候选，给精排留余量
    pool = max(top_k * 4, 10) if rerank else max(top_k * 3, top_k)
    res = col.query(query_embeddings=[q_vec], n_results=min(pool, total))
    vec_ranks = {}
    for i, m in enumerate(res["metadatas"][0]):
        # col.get 的顺序与 query 返回的 id 顺序不一定一致，用 idx+source 对齐
        vec_ranks[_doc_key(m["source"], m["idx"])] = i + 1

    if mode == "keyword":
        vec_ranks = {}

    # ---- 2. BM25 召回 ----
    bm_ranks = {}
    if mode in ("keyword", "hybrid"):
        bm = BM25Okapi([_tokenize(d) for d in all_docs["documents"]])
        scores = bm.get_scores(_tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[: pool]
        for rank, i in enumerate(order, 1):
            m = all_docs["metadatas"][i]
            bm_ranks[_doc_key(m["source"], m["idx"])] = rank

    # ---- 3. RRF 融合（粗排） ----
    if mode == "vector":
        final = vec_ranks
    elif mode == "keyword":
        final = bm_ranks
    else:
        K = 60  # RRF 平滑常数，越大越弱化头部排名优势
        final = {}
        keys = set(vec_ranks) | set(bm_ranks)
        for k in keys:
            final[k] = (1.0 / (K + vec_ranks[k]) if k in vec_ranks else 0.0) + (
                1.0 / (K + bm_ranks[k]) if k in bm_ranks else 0.0
            )

    # ---- 4. Rerank 精排 ----
    index_by_key = {}
    for doc, m in zip(all_docs["documents"], all_docs["metadatas"]):
        index_by_key[_doc_key(m["source"], m["idx"])] = (doc, m)

    ranked = sorted(final.items(), key=lambda x: -x[1])[:pool]
    cand = [(k, s, index_by_key[k]) for k, s in ranked]

    if rerank and len(cand) > top_k:
        order = llm_client.rerank(query, [d for _, _, (d, _) in cand], top_n=top_k)
        cand = [cand[i] for i in order]
    else:
        cand = cand[:top_k]

    out = []
    for rank, (key, score, (doc, m)) in enumerate(cand, 1):
        out.append({
            "text": doc,
            "source": m["source"],
            "idx": m["idx"],
            "rank": rank,
            "score": round(score, 5),
        })
    return out


def _doc_key(source: str, idx: int) -> str:
    return f"{source}#{idx}"


# ---------- 状态（最重要的调试函数） ----------
def stats() -> dict:
    if not os.path.exists(DB_DIR):
        return {"exists": False, "total": 0}
    chroma = chromadb.PersistentClient(path=DB_DIR)
    col = chroma.get_or_create_collection(name=COLLECTION)
    total = col.count()
    per_source = {}
    if total:
        got = col.get(include=["metadatas"])
        for m in got["metadatas"]:
            per_source[m["source"]] = per_source.get(m["source"], 0) + 1
    return {
        "exists": True,
        "total": total,
        "per_source": per_source,
        "chunk_size": CHUNK_SIZE,
        "overlap": CHUNK_OVERLAP,
        "top_k": TOP_K,
        # 关键警告：库比top_k 还小，说明检索形同虚设
        "degenerate": total > 0 and total <= TOP_K,
    }


if __name__ == "__main__":
    import sys

    force = "--rebuild" in sys.argv
    build(force=force)

    s = stats()
    print("\n[库状态]")
    print(f"  总块数   : {s['total']}")
    print(f"  各文件   : {s.get('per_source')}")
    print(f"  参数     : CHUNK_SIZE={s['chunk_size']} OVERLAP={s['overlap']} TOP_K={s['top_k']}")
    if s.get("degenerate"):
        print(f"  ⚠️警告  : 库总块数 {s['total']} <= TOP_K {s['top_k']}，"
              f"检索会退化成「全量注入」，请加大语料或调小 CHUNK_SIZE")