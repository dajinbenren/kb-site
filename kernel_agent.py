"""
kernel_agent.py —— 内核 B：Function Calling Agent
=================================================
手写 agent loop 的本质只有一句话：
    模型说「我要调哪个工具、传什么参数」→ 我们真的执行 → 结果以 role=tool
    回填进 messages → 模型下一轮才看得见结果并继续推理。

模型不能自己跑代码，这是安全边界，也是必须由我们执行的原因。

对比内核 A（RAG）：
    RAG   = 单轮，把 top-k 塞进去，模型一次性作答，不会翻第二页
    Agent = 多轮，模型自己决定查什么、算什么，证据不足会换关键词再查

定位差异：上一个问答「紧急发布审批」单轮 RAG 答「未提及」（答案排第 5，
被TOP_K=4 挡掉）。Agent 遇到同样情况会换个关键词重查，这就是多步取证。

三个真实工程防护（不是玩具代码，面试可讲）：
1. max_turns     防死循环，模型抽风时强制收尾。
2. 工具结果缓存   同一工具 + 同一参数不重复执行，直接复用，省 token 也省延迟。
3. 重复调用熔断   连续两轮调用完全相同的工具（且无新工具出现）就停止并提示，
                 防止模型空转烧钱。

面试话术：
    "我们手写了 function-calling agent loop，把 RAG 检索降级成一个可被调用的
     工具挂进去，形成 RAG-as-Tool。多步取证解决了单轮 RAG 召回不全的问题。
     实际用时补了三项防护：轮次上限、工具结果缓存、重复调用熔断，
     避免模型空转导致成本失控。"
"""
import json

import llm_client
import vector_store

MAX_TURNS = 6

AGENT_SYSTEM_PROMPT = """你是星辰科技的内部助理。

工作方式：
1. 任何涉及公司制度、规定、流程、技术规范的问题，都必须先用 search_kb 取证，禁止凭记忆回答。
2. 如果一次检索没找到关键依据，换一个关键词再检索一次，不要直接放弃。
3. 需要把条款里的标准代入具体数字时，用 calculate 计算，不要自己口算。
4. 依据不足时明确说明「知识库中未提及」，绝不编造条款、编号或数字。
5. 最终回答要给出结论 + 依据的文件名与条款编号。"""


# ============================================================
# 工具实现
# ============================================================
def _tool_search_kb(keyword: str, top_k: int = 3) -> str:
    hits = vector_store.search(keyword, top_k=top_k)
    if not hits:
        return "未检索到相关内容。"
    return "\n\n".join(
        f"【{h['source']} 第{h['idx']}块】\n{h['text']}" for h in hits
    )


def _tool_calculate(expression: str) -> str:
    try:
        return str(round(eval(expression, {"__builtins__": {}}, {}), 2))
    except Exception as e:
        return f"算不出来：{e}"


# ============================================================
# 工具说明书（JSON Schema）—— 模型靠这个决定调什么、传什么
# ============================================================
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_kb",
            "description": "检索星辰科技内部知识库（员工手册、IT运维手册、技术规范）。涉及公司制度、流程、标准、审批、技术规定时必须调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "检索关键词，如「紧急发布审批」「年假天数」「报销流程」",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回条数，默认 3；找不到时可加大到 5",
                    },
                },
                "required": ["keyword"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "计算算术表达式。需要把制度标准代入具体数字时调用，例如 4*500。",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "算术表达式，如 (5600-2000)/5600",
                    }
                },
                "required": ["expression"],
            },
        },
    },
]

FUNC_MAP = {
    "search_kb": _tool_search_kb,
    "calculate": _tool_calculate,
}


# ============================================================
# Agent loop
# ============================================================
def run(question: str, use_context: list[dict] | None = None, trace: bool = True) -> dict:
    messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]
    if use_context:
        messages.extend(use_context[-6:])

    messages.append({"role": "user", "content": question})

    steps: list[dict] = []          # 前端要展示的工具轨迹
    cache: dict[str, str] = {}      # 防护2：工具结果缓存
    last_call_key = None# 防护3：熔断用
    repeats = 0

    for turn in range(1, MAX_TURNS + 1):
        msg = llm_client.chat(messages, tools=TOOLS)

        # —— 分支 1：模型不想调工具了 → 这是最终答案 ——
        if not msg.get("tool_calls"):
            return {
                "answer": msg["content"],
                "steps": steps,
                "turns": turn,
                "mode": "agent",
            }

        messages.append({k: v for k, v in msg.items() if k in ("role", "content", "tool_calls")})

        for tc in msg["tool_calls"]:
            name = tc["function"]["name"]
            args = json.loads(tc["function"]["arguments"])
            call_key = f"{name}:{json.dumps(args, sort_keys=True, ensure_ascii=False)}"

            # 防护3：连续重复同一调用 → 熔断
            if call_key == last_call_key:
                repeats += 1
            else:
                repeats = 0
            last_call_key = call_key
            if repeats >= 2:
                msg_txt = ("检测到重复调用同一工具且无新进展，已停止重试。"
                           "建议换用更精确的关键词，或直接参考已有资料作答。")
                messages.append({"role": "user", "content": msg_txt})
                steps.append({"tool": name, "args": args, "result": "（熔断：重复调用）",
                              "cached": False, "turn": turn})
                if trace:
                    print(f"  [熔断] {name}{args}")
                continue

            # 防护2：缓存命中直接复用，不重复执行
            if call_key in cache:
                result = cache[call_key]
                cached = True
            else:
                result = str(FUNC_MAP[name](**args))
                cache[call_key] = result
                cached = False

            steps.append({"tool": name, "args": args, "result": result,
                          "cached": cached, "turn": turn})
            if trace:
                tag = f"{name}{args}"
                print(f"  [第{turn}轮] 🔧 {tag}{' (缓存)' if cached else ''}")

            # ★ 灵魂一行：结果回填，模型下一轮才看得见
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result,
            })

    #轮次用尽：强行收尾，让模型基于现有信息作答
    messages.append({"role": "user",
                     "content": "请停止继续调用工具，基于目前已获取的资料直接给出最终答案。"})
    msg = llm_client.chat(messages)
    return {"answer": msg["content"], "steps": steps, "turns": MAX_TURNS, "mode": "agent"}


if __name__ == "__main__":
    import sys

    vector_store.build()

    qs = sys.argv[1:] or [
        "紧急发布需要什么审批？",
        "我出差住了一线城市酒店 4 晚一共 5600 元，制度下能报多少？超标比例多少？",
    ]
    for q in qs:
        print("\n" + "=" * 64)
        print("❓", q)
        print("-" * 64)
        r = run(q)
        for s in r["steps"]:
            prev = s["result"][:70].replace("\n", " ")
            print(f"  🔧 {s['tool']}({s['args']}){' [缓存]' if s['cached'] else ''}")
            print(f"     ↳ {prev}{'…' if len(s['result']) > 70 else ''}")
        print(f"\n💬 {r['answer']}")
        print(f"\n[共{r['turns']} 轮，{len(r['steps'])} 次工具调用]")