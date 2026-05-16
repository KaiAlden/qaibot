# QAibot 中医体质 RAG 问答系统

## 项目简介

QAibot 是一个面向中医体质、饮食调理和养生建议的 RAG 问答项目。项目基于本地知识库和大模型能力，为用户提供体质识别、体质特征说明、饮食建议、起居建议、情绪调节、运动调理等问答服务。

项目主要解决以下问题：

- 将中医九种体质、地区、季节、节气、饮食和调理建议结构化后接入问答系统。
- 支持多轮对话中的体质、地区、季节等上下文继承。
- 支持非流式 `/chat` 和流式 `/chat/stream` 两种回答方式。
- 支持本地 OpenAI-compatible 大模型服务，也可以切换为云端大模型 API。
- 支持模型思考过程与正式回答分流，流式输出中 `thinking` 和 `delta` 分开返回。
- 预留天气、音乐、联网搜索、MCP 等外部工具调用接口。

当前项目仍然是自研编排的 FastAPI RAG 服务，没有迁移到 LangChain 或 LangGraph。

## 技术栈

主要技术与依赖如下：

- Python：项目主要开发语言。
- FastAPI：提供 HTTP API 服务。
- Uvicorn：运行 ASGI 服务。
- Pydantic：定义请求、响应和内部数据结构。
- OpenAI SDK：调用 OpenAI-compatible LLM 与 Embedding 服务，兼容本地 vLLM / 私有模型服务。
- Qdrant：向量数据库，用于体质识别知识和调理建议知识检索。
- MySQL：保存用户会话状态和多轮对话历史。
- PyMySQL：连接 MySQL。
- pandas / openpyxl：读取 Excel 知识源并构建知识库。
- python-dotenv：读取 `.env` 配置文件。
- SSE：`/chat/stream` 使用 Server-Sent Events 实现流式输出。

核心能力模块：

- RAG 检索：基于 Qdrant 的向量检索和 payload 过滤。
- 意图路由：快速规则优先，LLM Router 兜底。
- 思考分流：解析 `<think>`、`</think>`、`<answer>`、`</answer>` 等模型输出协议。
- 工具预留：通过 `ToolExecutor` 预留天气、音乐、Web Search、MCP 工具入口。

## 目录结构

```text
QAibot/
  app/
    main.py
      FastAPI 入口，定义 /health、/chat、/chat/stream 接口。

    config.py
      读取 .env 配置，管理 LLM、Embedding、Qdrant、MySQL、流式输出等参数。

    schemas.py
      定义 ChatRequest、ChatResponse、SessionState、RoutedTask、ToolCall 等数据结构。

    session_store.py
      MySQL 会话存储，负责保存 user_id + conversation_id 维度的会话状态和历史。

    domain/
      constants.py
        中医体质、地区、季节、节气、建议类型等领域常量。

      normalizers.py
        体质、地区、节气、建议类型等归一化逻辑。

    nlp/
      general_intent.py
        通用寒暄、能力介绍、感谢、告别等简单意图处理。

      intent_parser.py
        旧版规则意图解析，目前主要作为槽位补全使用。

      task_router.py
        当前主路由器，负责判断 tcm_health、weather、music、web_search 等任务路由。

      clarification.py
        判断是否需要向用户追问补充信息。

    rag/
      chunk_builder.py
        读取 Excel / TXT 知识源并生成知识块。

      qdrant_store.py
        Qdrant 与 Embedding 封装，负责建库、写入和检索。

      constitution_identifier.py
        体质识别模块。

      retriever.py
        饮食、起居、情绪、运动等调理建议检索模块。

      answer_generator.py
        根据检索资料、会话上下文和运行上下文调用大模型生成回答。

      thinking.py
        模型思考过程解析器，负责将 thinking 和 answer 拆分。

    services/
      chat_service.py
        问答主链路编排，串联路由、识别、检索、生成、流式输出和会话保存。

    tools/
      executor.py
        外部工具调用预留入口，目前返回 not_configured。

  scripts/
    build_index.py
      构建知识库并写入 Qdrant。

    intent_demo.py
      意图识别调试脚本。

  data/
    项目知识源文件目录。

  requirements.txt
    Python 依赖列表。

  README.md
    项目说明文档。
```

## 开发规范

### 代码风格

- 使用 Python 3 类型注解，函数入参和返回值尽量显式声明类型。
- API 入参、响应和跨模块数据结构优先使用 Pydantic Model。
- 业务逻辑按层拆分，避免把路由、检索、生成、存储逻辑全部写在接口函数里。
- RAG 主流程放在 `app/services/chat_service.py` 中编排。
- LLM 调用相关逻辑集中在 `app/rag/answer_generator.py` 和 `app/nlp/task_router.py`。
- 外部工具调用统一从 `app/tools/executor.py` 扩展。

### 命名规则

- 文件名、函数名、变量名使用小写加下划线，例如 `chat_service.py`、`load_settings`。
- 类名使用大驼峰，例如 `ChatService`、`TaskRouter`、`ThinkingStreamParser`。
- 配置项使用大写下划线，例如 `LLM_BASE_URL`、`QDRANT_URL`、`MYSQL_DATABASE`。
- API 字段使用小写下划线，例如 `user_id`、`conversation_id`、`runtime_context`。

### 业务约定

- `/chat` 返回完整回答，`answer` 中只保留正式回答。
- `/chat/stream` 使用 SSE 输出：
  - `status`：阶段状态。
  - `thinking`：模型思考过程或思考摘要。
  - `delta`：正式回答增量。
  - `done`：结束事件和元信息。
  - `error`：异常事件。
- `last_advice_types` 用于保存最近一次建议类型，不再使用 `last_advice_type`。
- 中医养生、体质、饮食、调理、节气健康类问题默认走 RAG。
- 天气、音乐、非中医开放知识查询走外部工具预留通道。
- 路由优先使用快速规则，模糊问题再调用 LLM Router。
- 思考过程展示由 `THINKING_DISPLAY_MODE` 控制，生产环境建议使用 `summary` 或 `off`。

## 如何运行

### 1. 创建并进入 Python 环境

项目当前常用 conda 环境名为 `QAibot-env`：

```powershell
conda activate QAibot-env
```

如果还没有该环境，可以自行创建：

```powershell
conda create -n QAibot-env python=3.10
conda activate QAibot-env
```

### 2. 安装依赖

```powershell
pip install -r requirements.txt
```

如果 MySQL 使用 `caching_sha2_password` 或 `sha256_password` 认证方式，还需要安装：

```powershell
pip install cryptography
```

### 3. 配置环境变量

在项目根目录创建 `.env` 文件，并按实际环境填写配置。

最小配置示例：

```env
# LLM
LLM_PROVIDER=openai
LLM_API_KEY=local-key
LLM_BASE_URL=http://127.0.0.1:8000/v1
LLM_MODEL=/path/to/your/local/model
LLM_TEMPERATURE=0.2
LLM_REQUEST_TIMEOUT=90
ROUTER_LLM_TIMEOUT=6

# Thinking
THINKING_DISPLAY_MODE=summary
THINKING_START_TAG=<think>
THINKING_END_TAG=</think>
THINKING_ANSWER_START_TAG=<answer>
THINKING_ANSWER_END_TAG=</answer>
THINKING_SUMMARY_MAX_CHARS=600
THINKING_STREAM_BUFFER_CHARS=1200

# Embedding
EMBEDDING_PROVIDER=openai
EMBEDDING_API_KEY=your_embedding_key
EMBEDDING_BASE_URL=https://api.example.com/v1
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DIM=1024

# Qdrant
QDRANT_URL=http://127.0.0.1:6333
QDRANT_API_KEY=
QDRANT_CONSTITUTION_COLLECTION=tcm_constitution_knowledge
QDRANT_ADVICE_COLLECTION=tcm_advice_knowledge
QDRANT_DISTANCE=Cosine

# MySQL
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=your_password
MYSQL_DATABASE=qaibot
MYSQL_CHARSET=utf8mb4

# Session
SESSION_TTL_DAYS=30
SESSION_HISTORY_TURNS=12

# RAG
DEFAULT_AREA=华北
DEFAULT_TOP_K=5
CONSTITUTION_IDENTIFY_TOP_K=4
DIET_PRINCIPLE_TOP_K=1
SUGGESTION_TOP_K=1
SUGGESTION_PER_TYPE_TOP_K=1
GENERAL_SUGGESTION_TOP_K=4
```

说明：

- 如果本地大模型服务不校验 API Key，`LLM_API_KEY` 也建议填写一个占位值，例如 `local-key`。
- `LLM_BASE_URL` 需要指向 OpenAI-compatible 服务的 `/v1` 地址。
- `EMBEDDING_DIM` 必须和实际 Embedding 模型输出维度一致。
- `THINKING_DISPLAY_MODE` 可选值为 `summary`、`raw`、`off`。

### 4. 构建知识库

先 dry-run 验证分块结果：

```powershell
conda run -n QAibot-env python scripts\build_index.py --dry-run
```

确认无误后写入 Qdrant：

```powershell
conda run -n QAibot-env python scripts\build_index.py
```

### 5. 启动 API 服务

```powershell
conda run -n QAibot-env uvicorn app.main:app --host 0.0.0.0 --port 8000
```

浏览器或接口工具访问健康检查：

```http
GET http://localhost:8000/health
```

### 6. 调用非流式接口

```http
POST http://localhost:8000/chat
Content-Type: application/json
```

请求体示例：

```json
{
  "user_id": "user1",
  "conversation_id": "conv001",
  "message": "湿热体质的特征是什么？",
  "runtime_context": {
    "location": "杭州",
    "current_time": "2026-05-14 10:00:00",
    "solar_term": "小满",
    "weather": "晴"
  }
}
```

### 7. 调用流式接口

```http
POST http://localhost:8000/chat/stream
Content-Type: application/json
Accept: text/event-stream
```

请求体示例：

```json
{
  "user_id": "user1",
  "conversation_id": "conv001",
  "message": "湿热体质夏季饮食怎么调理？",
  "runtime_context": {
    "location": "杭州",
    "solar_term": "小满",
    "weather": "晴"
  }
}
```

流式事件示例：

```text
event: status
data: {"text": "正在理解你的问题..."}

event: thinking
data: {"text": "思考摘要..."}

event: delta
data: {"text": "正式回答内容..."}

event: done
data: {"need_clarification": false, "route": "tcm_health"}
```
