# kb-site

> 配套项目：`kb-site/`（本地知识库问答网站）
> 6 个文件，一条数据流串起来：**用户提问 → 检索原文 → 模型作答 → 存库返回**。
> 本导读按「层」逐个文件讲：这个文件干什么、里面写了什么、每块代码起什么作用。

---

## 先记住整体结构（4 层）

| 层 | 文件 | 一句话定位 |
|---|---|---|
| 语料 | `data/*.md` | 知识库原文（3 份文档） |
| 底座 | `llm_client.py`<br>`vector_store.py` | 调模型 / 存和搜向量，**不懂问答** |
| 内核 | `kernel_rag.py`<br>`kernel_agent.py` | 产生智能：单轮抄答案 / 多步调工具 |
| 服务 | `main.py` | 把内核包成 HTTP 接口 + 存会话 |
| 前端 | `static/index.html` | 能点、能看、能切换的网页 |

**核心原则**：先跑通内核再包 API。反过来会把「内核 bug」误判成「接口 bug」。

---

## 文件 1：`llm_client.py` —— 大模型客户端

**干什么**：把「怎么调大模型」收拢到一处。别人只管调函数，不用管哪家模型、地址怎么拼。

**像什么**：公司的总机。想找模型，转总机就行，不用自己知道号码。

### 三个函数

**① `chat()` —— 对话**
```python
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)  # 关键一行
```
- **调千问却写 OpenAI**：千问提供「OpenAI 兼容模式」，接口格式一样，只换 `BASE_URL` 就通。
- 这叫「供应商可替换性」——换模型只改配置，业务代码不动。
- `tool_choice="auto"` 是 Function Calling 总开关：传了 `tools`，模型就获得「可以调工具」的能力。
- `temperature=0.2` 调低，要稳定准确不要创意。

**灵魂一行**：
```python
return resp.choices[0].message.model_dump(...)
```
返回的字典里**可能有 `content`，也可能有 `tool_calls`**。这个「不确定」就是 agent loop 存在的原因。

**② `embed()` —— 向量化**
- 把文本变 1024 个浮点数，语义相近的文本数字方向也相近。
- **真实坑**：千问 embedding 单批最多 10 条，传 14 条报 `400 batch size is invalid` → 自动分批。
- **顺序必须严格对应**：上层拿向量数组去和 Chroma 原文配对，顺序错位 = 全库对齐崩掉，不报错只答错。

**③ `rerank()` —— 精排**
- **向量检索 = 快递员按地址打分**（快但只看大概）；**rerank = 打开包裹逐字看**（慢但准）。
- 技术上：向量是「双塔」（query/doc 各自编码无交互），rerank 把两者拼一起交叉编码。
- **真实坑**：OpenAI SDK 没有 rerank 方法，必须自己发 HTTP。
- **真实坑**：千问两个排序模型接口路径不同且都不在 `/compatible-mode/v1`：
  - `qwen3-rerank` → `/compatible-api/v1/reranks`（扁平结构）
  - `gte-rerank-v2` → `/api/v1/services/rerank/text-rerank/text-rerank`（input/parameters 嵌套）
- **降级兜底**：精排失败保持原顺序，绝不做单点故障。

---

## 文件 2：`vector_store.py` —— 向量库封装

**干什么**：把「文档怎么变成能搜的东西」包起来。对外三个动作：`build()` 建库、`search()` 检索、`stats()` 体检。

### 三个参数（面试要能讲清）

```python
CHUNK_SIZE = 450      # 每块最多 450 字
CHUNK_OVERLAP = 80    # 相邻块重叠 80 字
TOP_K = 4             # 送 4 块给模型
```

- **CHUNK_SIZE**：切太小，一句话被腰斩；切太大，90% 废话还挤爆上下文。
- **CHUNK_OVERLAP**：防「切腰」。`step = 450 - 80 = 370`，窗口宽 450 但只挪 370，相邻必有 80 字重叠。
- **TOP_K ≠ 取几条**，而是「你愿意给模型多少证据」。大了稀释+烧 token，小了漏关键条款。

### 切分 `load_and_split()`
- 先按 Markdown 标题分节，**标题一并存进块里**——否则「第九条」孤立无上下文，模型看不懂是哪个公司的第九条。
- 过长的节再按字符窗口滑切。

### 建库 `build()`
- 切分 → 向量化 → 清空旧库 → 写入。
- **真实坑**：`shutil.rmtree` 删目录被沙箱拦 → 改用 `chroma.delete_collection()`。

### 检索 `search()` —— 全文技术含量最高的一段，四步：

**第 1 步 向量召回**：问题转向量，找方向最近的块。
**第 2 步 BM25 召回**：字面词频匹配。
- 为什么需要：纯向量下「紧急发布审批/流程/例外发布」返回同一批块——**换词不改变向量邻域**。
- 中文分词 `_tokenize`：中文没空格，用 **bigram** 把「紧急发布审批」切 `紧急/急发/发布/布审/审批`，才能命中。
- **真实坑**：手写 BM25 按整词匹配 + IDF 饱和，效果差 → 换 `rank_bm25` + bigram。

**第 3 步 RRF 融合**：
```python
final[k] = 1.0/(60+vec_rank[k]) + 1.0/(60+bm_rank[k])
```
- **只用排名不用分数**，天然绕开余弦分与 BM25 分量纲不可比。
- **真实坑**：先试「归一化加权 0.6/0.4」，量纲不同拍脑袋定权重不可靠 → 改 RRF。

**第 4 步 rerank 精排**：粗排故意多捞（`top_k*4`），精排再选 top-k。
- 生产链路：**向量+BM25 粗排 → RRF → 召回 16 块 → rerank → top-3 → 送 LLM**。

### 体检 `stats()` —— 最值钱的经验
```python
"degenerate": total > 0 and total <= TOP_K
```
**库总块数 ≤ TOP_K = 检索退化成全量注入，压根没筛选。**
上一轮实测：2511 字只切 2 块，`TOP_K=3`，纯 RAG 和 Agent 都满分——**因为根本没有检索**。
**上手任何 RAG 第一件事：打印 chunk 数和 TOP_K。**

---

## 文件 3：`kernel_rag.py` —— 内核 A（RAG 问答）

**干什么**：把搜到的原文塞进 prompt，让模型照着抄，不许编。**单轮、开卷考试。**

### 三个重点

**① 防幻觉 prompt**：
```python
"参考资料里没有明确写的，直接回答「知识库中未提及该信息」，绝不允许推测"
```
作用不是让模型变聪明，是**给模型一条「允许认输」的退路**，否则它会硬编。
- **假阴性**（证据没捞到 → 说未提及）= 检索的锅，诚实可接受。
- **真幻觉**（证据捞到 → 还编数字）= 模型的锅，靠这条 prompt 压。

**② 核心三行**：`search()` 搜 → 拼成 `【来源：...】` 上下文 → 塞进 prompt。
- 来源标记写进 prompt，既让模型引用对得上，又返回前端展示——**可溯源**。

**③ 多轮**：`use_context[-6:]` 只带最近 6 轮，防 token 膨胀；历史放参考资料之前。

---

## 文件 4：`kernel_agent.py` —— 内核 B（Function Calling Agent）

**干什么**：手写 while 循环，模型自己决定「先查什么、再算什么」，多步取证。

### 五个要点

**① 工具实现**：`_tool_search_kb` 内部调 `vector_store.search()` = **把 RAG 降级成工具**（RAG-as-Tool）。
`_tool_calculate` 用 `eval(..., {"__builtins__": {}})` 禁内置函数防注入。

**② 工具说明书 TOOLS**：模型看不到 Python 代码，只看这段 JSON。description 措辞讲究——「涉及制度时必须调用」「不要自己口算」，逼模型按规矩来。

**③ FUNC_MAP**：
```python
FUNC_MAP = {"search_kb": _tool_search_kb, "calculate": _tool_calculate}
```
**安全边界**：模型只能喊名字，由后端白名单查出真函数执行。模型不能直接指定「执行哪段代码」。

**④ agent loop**：
- **灵魂一行**：结果用 `role="tool"` 回填 messages，模型下一轮才看得见。不回填 = 白调。
- **三个工程防护**（和玩具 Agent 的分水岭）：
  - `max_turns=6` 防死循环
  - 缓存 `cache` 同一调用不重复执行
  - 熔断 `repeats>=2` 连续重复调用就打断
- **缓存 vs 熔断**：缓存省「重复执行」，熔断省「无效轮次」。

**⑤ 兜底**：轮次用尽强制收尾，保证函数一定返回。

---

## 文件 5：`main.py` —— FastAPI 服务层（壳）

**干什么**：把内核包成 HTTP 接口 + 存会话。**不产生智能**，只做「收请求→校验→转发→存库→返回」。

### 四个重点

**① SQLite**：`sessions`（会话）+ `messages`（消息）两张表。
- 为什么不用内存 list：服务重启就没了。
- 为什么不上 Postgres：本地小项目 SQLite 免安装够用。

**② lifespan**：启动时建库（而不是第一次提问才建，否则首个用户等十几秒）。
- **真实坑**：旧写法 `on_event` 已弃用 → 改用 `lifespan` 异步上下文管理器。

**③ Pydantic 校验**：`question` 空、`mode` 非法、`top_k` 超限 → 直接 422，脏数据挡在门口。

**④ `/api/ask` 心脏**：
```
前端 → HTTP → main.py → 内核 → 底座 → 千问 → 结果 → 存 SQLite → 返回前端
```

---

## 文件 6：`static/index.html` —— 前端

**干什么**：给内核套能点能看的界面。纯 HTML+JS，不上框架（杀鸡不用牛刀）。

### 三个亮点

**① 双模式切换**：`mode: 'rag' | 'agent'` 传给后端换内核。
**② 工具轨迹可视化**：Agent 每一步「调了什么工具、传了什么参、返回什么」画出来——Agent 的**可观测性**。
**③ 引用来源展示**：RAG 的「可溯源」落到界面，答案能核对原文。

---

## 一条完整数据流（把 6 个文件串起来）

```
用户点网页问「报销多少」
   ↓ index.html 发 POST /api/ask
   ↓ main.py 校验 + 取历史
   ↓ kernel_agent.py 模型说「我要查住宿标准」
   ↓ vector_store.py 混合检索 + rerank 捞出原文
   ↓ llm_client.py 调千问，模型说「我要算 4*500」
   ↓ kernel_agent.py 调 calculate 算出 2000，回填
   ↓ 模型给最终答案 → main.py 存 SQLite → 返回前端
   ↓ index.html 画出答案 + 工具轨迹 + 引用来源

