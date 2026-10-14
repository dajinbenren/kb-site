# 星辰知识库助手 · 本地小网站

一个从零搭起来的本地 AI 应用：**FastAPI + RAG 检索增强 + Function Calling 工具调用 + SQLite 会话持久化 + 原生 HTML 前端**。

---

## 一、怎么跑起来

```bash
cd kb-site

# 1. 确认 .env 里有key（已配好）
#    DASHSCOPE_API_KEY=sk-xxx

# 2. 装依赖
D:/DailySoftWare/Python/python.exe -m pip install fastapi uvicorn openai python-dotenv chromadb rank_bm25 httpx

# 3. 启动（--reload 让改代码自动重启）
D:/DailySoftWare/Python/python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

打开 **http://127.0.0.1:8000** 就是网站；**http://127.0.0.1:8000/docs** 是自动生成的 API 文档。

改完语料后，点页面右上角「重建索引」即可，不用重启服务。

---

## 二、四层架构

| 层 | 文件 | 职责 |
|---|---|---|
| 第1 层 语料 | `data/*.md` | 3 份文档：员工手册 / IT运维手册 / 技术规范 |
| 第 2 层 底座 | `llm_client.py`<br>`vector_store.py` | 千问客户端（对话/向量化/精排）<br>切分+建库+混合检索+rerank |
| 内核 | `kernel_rag.py`<br>`kernel_agent.py` | A：RAG 单轮检索问答<br>B：手写 function-calling agent loop |
| 第 3 层 服务 | `main.py` | FastAPI 接口 + lifespan 建库 + SQLite 会话 |
| 第 4 层 前端 | `static/index.html` | 双模式切换、工具轨迹可视化、引用来源、历史会话 |

**搭建顺序的关键**：先跑通内核（`python kernel_rag.py`），再包 API。反过来会把「内核 bug」误判成「接口 bug」。

---

## 三、检索链路（本项目最值钱的部分）

```
用户问题
   ↓
向量召回(text-embedding-v3, top-12)  ┐
                    ├─ RRF 融合 ─→ 粗排 top-10
BM25 召回(rank_bm25, 中文 bigram)    ┘
   ↓
Rerank 精排(gte-rerank-v2) → top-3
   ↓
拼 prompt + 防幻觉约束 → qwen-plus 生成
```

三个参数（在 `vector_store.py` 顶部）：

| 参数 | 值 | 含义 |
|---|---|---|
| `CHUNK_SIZE` | 450 | 每块字符数上限 |
| `CHUNK_OVERLAP` | 80 | 块间重叠，防边界断句 |
| `TOP_K` | 4 | 最终送给 LLM 的块数 |

**必看**：先跑 `python vector_store.py` 看 `stats()`。它会告诉你库里有几块，
以及当块数`<=` TOP_K 时检索已退化成「全量注入」——那时你在测的是一个假检索系统。

---

## 四、接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 网站首页 |
| GET | `/api/health` | 模型名 + 向量库状态 |
| POST | `/api/ask` | 提问，`mode` 为 `rag` 或 `agent` |
| GET | `/api/sessions` | 历史会话列表 |
| GET | `/api/sessions/{sid}` | 某会话完整消息 |
| POST | `/api/rebuild` | 强制重建索引 |

```bash
# 用文件传参（Git Bash 下中文 JSON 直接内联会error parsing body）
curl -X POST http://127.0.0.1:8000/api/ask \
  -H "Content-Type: application/json" \
  -d @req_agent.json
```

---

## 五、踩坑记录（全部真实踩过）

| # | 现象 | 原因 | 修复 |
|---|---|---|---|
| 1 | embedding 报 `batch size is invalid` | 千问 embedding 单批上限 10 条 | `llm_client.embed` 自动分批 |
| 2 | Chroma `Validation error: name` | 1.5+ 要求集合名 ≥3 字符 | `"kb"` → `"kb_docs"` |
| 3 | `shutil.rmtree` 被沙箱拦截 | 容器/只读环境常见 | 改用 `chroma.delete_collection()` |
| 4 | Agent 重复调同一工具空转 | 模型抽风 | 工具结果缓存 + 连续重复熔断 |
| 5 | 换关键词检索结果不变 | 纯向量检索的固有局限 | 上 BM25 混合 + RRF 融合 |
| 6 | 手写 BM25 效果很差 | 整词匹配不识别中文短语 | 换`rank_bm25` + 中文 bigram 分词 |
| 7 | 归一化线性加权不可靠 | 余弦分与 BM25 分量纲不同 | 改用 RRF，**只用排名不用分数** |
| 8 | `OpenAI` 对象无 `rerank` 方法 | SDK 不含该能力 | 直接发 HTTP 请求 |
| 9 | rerank 报 404 | 两个模型端点路径不同 | 见下方说明 |

**千问 rerank 两个端点（都试一遍取先成功的）**：
- `qwen3-rerank` → `POST /compatible-api/v1/reranks`（扁平结构）
- `gte-rerank-v2` → `POST /api/v1/services/rerank/text-rerank/text-rerank`（`input`/`parameters` 嵌套）

**都是 `/compatible-mode/v1` 之外的路径**。精排失败时自动降级为粗排原顺序——
精排是锦上添花，不能做单点故障。

---

## 六、面试话术

**RAG vs Agent 的本质区别**
> RAG 是开卷考试但只会抄：把 top-k 段落一次塞进去，模型单轮作答，不会再翻第二页。
> Agent 是自己决定先查什么再算什么，证据不足会换关键词重查，多步取证。
> 真实工程里几乎都是 **RAG-as-Tool**——把检索降级成工具塞进 agent loop，不是二选一。

**为什么需要 rerank**
> 向量检索是双塔粗排，query 和 doc 各自编码、无交互，快但粗；
> rerank 把 query 和每个 doc 拼起来做交叉编码，逐对深度交互，慢但准。
> 生产链路就是粗排召回 30 条→ 精排 top-3 → 送 LLM。
> 本项目实测：粗排时真答案排第 4 名被 TOP_K 挡在门外，加 rerank 后进 top-2。

**假阴性vs 幻觉**
> 检索没召回 → 模型答「未提及」= **假阴性**，是检索的锅，靠 rerank/加大 TOP_K 解决。
> 证据充足仍编数字 = **真幻觉**，靠约束 prompt 解决。
> 本项目「紧急发布审批」一题两种都出现过，是很好的对照案例。

**为什么接口层用同步 def 而不是 async**
> 我们用的 openai SDK 是同步的，写 `async def` 反而会阻塞事件循环。
> FastAPI 对普通 `def` 会自动丢到线程池，正是想要的效果。
> 异步的价值在于「等待网络时不让出CPU」，同步 SDK 做不到，得换httpx async。

**agent loop 的工程防护**
> 轮次上限防死循环；工具结果缓存省 token；重复调用熔断防空转烧钱。
> 这些不是玩具代码，模型确实会重复调同一个工具拿回一样的结果。