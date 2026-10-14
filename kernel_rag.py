"""
kernel_rag.py —— 内核 A：RAG 检索增强问答
=========================================
五段式：检索 → 拼上下文 → 生成 → 附来源。

定位：单轮、开卷考试。模型只能「抄」你给它的那几块原文。
优点：稳、快、成本低，答案可溯源。
缺点：只会一次性作答，不会自己再翻一页；也不会算数。

面试话术：
    "RAG 是把检索结果作为上下文注入 prompt 的单轮方案。核心价值是把模型
     的回答锚定在私域文档上，避免幻觉并给出引用来源。它的瓶颈在检索——
     召回不全时模型只能答『未提及』，这是假阴性，不等于幻觉。工程上
     常见做法是加大TOP_K、加重叠，或改用 rerank 精排。"
"""
import vector_store
import llm_client

RAG_SYSTEM_PROMPT = """你是星辰科技内部知识库问答助手。

必须遵守：
1. 只依据「参考资料」回答，不编造资料之外的信息。
2. 参考资料里没有明确写的，直接回答「知识库中未提及该信息」，
   绝不允许推测、绝不允许用自己的常识补齐。
3. 引用条款时注明来源文件和条款编号。
4. 若问题需要计算，说明依据的条款，再给出计算过程。"""


def answer(question: str, top_k: int = None, use_context: list[dict] | None = None) -> dict:
    """RAG 问答主入口。

    use_context: 可传入历史对话（[{"role","content"}...]）做多轮问答。
    """
    k = top_k or vector_store.TOP_K
    hits = vector_store.search(question, top_k=k)
    context_text = "\n\n".join(
        f"【来源：{h['source']}｜第{h['idx']}块】\n{h['text']}" for h in hits
    )

    messages = [{"role": "system", "content": RAG_SYSTEM_PROMPT}]
    # 历史放在资料之前：让模型有多轮语境，但当前证据始终是最后也是最新的那块
    if use_context:
        messages.extend(use_context[-6:])  # 只带最近几轮，防止 token 膨胀
    messages.append(
        {"role": "user", "content": f"参考资料：\n{context_text}\n\n问题：{question}"}
    )

    msg = llm_client.chat(messages)
    return {
        "answer": msg["content"],
        "sources": [
            {"source": h["source"], "idx": h["idx"], "rank": h["rank"],
             "preview": h["text"][:60].replace("\n", " ")}
            for h in hits
        ],
        "mode": "rag",
    }


if __name__ == "__main__":
    import sys
    import json

    qs = sys.argv[1:] or [
        "入职满 3 年能享有多少天年假？",
        "P1 阻断级故障多久必须恢复？",
        "紧急发布需要什么审批？",
    ]
    vector_store.build()

    for q in qs:
        r = answer(q)
        print("\n" + "=" * 64)
        print("❓", q)
        print("-" * 64)
        print("💬", r["answer"])
        print("📚 引用：", "、".join(
            f"{s['source']}#{s['idx']}" for s in r["sources"]
        ))