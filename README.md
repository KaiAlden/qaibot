# QAibot TCM Constitution RAG

基于 Qdrant 的中医九体质饮食与调理 RAG 工程。系统使用 `data/体质饮食表.xlsx` 和 `data/体质症状补充映射.txt` 构建知识库，支持体质识别、饮食建议、调理建议、多轮追问、会话记忆，以及 `user_id + conversation_id` 级别的会话隔离。

## 1. 环境准备

项目运行在 conda 环境 `QAibot-env` 下：

```powershell
conda activate QAibot-env
pip install -r requirements.txt
```

复制配置模板并填写真实信息：

```powershell
Copy-Item .env.example .env
```

至少需要填写：

```env
LLM_API_KEY=
LLM_MODEL=
EMBEDDING_API_KEY=
QDRANT_URL=
QDRANT_API_KEY=
QDRANT_CONSTITUTION_COLLECTION=tcm_constitution_knowledge
QDRANT_ADVICE_COLLECTION=tcm_advice_knowledge
MYSQL_HOST=
MYSQL_PORT=
MYSQL_USER=
MYSQL_PASSWORD=
MYSQL_DATABASE=
```

如果 LLM 和 Embedding 使用同一个 OpenAI 兼容服务，可以只填 `LLM_API_KEY`，并按需填写 `LLM_BASE_URL`。

## 2. 构建知识库

先验证分块结果：

```powershell
conda run -n QAibot-env python scripts\build_index.py --dry-run
```

写入 Qdrant：

```powershell
conda run -n QAibot-env python scripts\build_index.py
```

当前数据的 dry-run 结果应为：

```text
Built 2011 chunks
- constitution_identify: 9
- diet_principle: 286
- suggestion: 1716
```

写入时会使用两个 Qdrant collection：

```text
tcm_constitution_knowledge
  存 constitution_identify，用于体质识别。

tcm_advice_knowledge
  存 diet_principle 和 suggestion，用于饮食原则与调理建议检索。
```

当前 Qdrant payload 只保留必要字段：

```text
constitution_identify:
  chunk_id, type, content, constitution

diet_principle / suggestion:
  chunk_id, type, content, area, season, constitution, suggestion_name
```

当前不会写入 `source`、`symptom_keywords`、`solar_term`。节气映射仍保留在代码里，只用于把用户输入或源数据中的节气归一到 `season`。

## 3. 启动 API

```powershell
conda run -n QAibot-env uvicorn app.main:app --host 0.0.0.0 --port 8000
```

健康检查：

```http
GET http://localhost:8000/health
```

聊天接口：

```http
POST http://localhost:8000/chat
Content-Type: application/json

{
  "user_id": "u001",
  "conversation_id": "conv001",
  "message": "我在广东，最近手脚冰凉，春季应该怎么吃？"
}
```

## 4. 会话存储与上下文

会话状态保存在 MySQL 表 `qaibot_sessions` 中。服务启动时会自动创建 `MYSQL_DATABASE` 和该表。

隔离规则：

- `user_id` 不同：完全隔离。
- `conversation_id` 不同：即使是同一用户，也视为不同窗口，互不共享当前体质、地区、季节等上下文。
- 同一 `user_id + conversation_id`：支持多轮追问和上下文继承。

相关配置：

```env
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=
MYSQL_PASSWORD=
MYSQL_DATABASE=qaibot
MYSQL_CHARSET=utf8mb4

SESSION_TTL_DAYS=30
SESSION_HISTORY_TURNS=12
```

说明：

- `SESSION_HISTORY_TURNS=12` 表示 prompt 和会话状态中默认最多保留最近 12 轮对话。
- 1 轮对话 = 1 条用户消息 + 1 条助手回答。
- `SESSION_TTL_DAYS=30` 只是 MySQL 旧会话清理策略，不代表把 30 天历史都放进上下文窗口。

## 5. RAG 核心模块

```text
app/rag/chunk_builder.py
  读取 Excel/TXT，生成 constitution_identify、diet_principle、suggestion 三类 chunk。

app/rag/qdrant_store.py
  Qdrant 适配层，负责建 collection、embedding、upsert、按 payload filter 检索。

app/rag/constitution_identifier.py
  体质识别模块，只检索 QDRANT_CONSTITUTION_COLLECTION。

app/rag/retriever.py
  饮食/调理知识检索模块，只检索 QDRANT_ADVICE_COLLECTION，并支持 fallback 降级。

app/rag/answer_generator.py
  基于检索资料、体质识别结果和最近对话历史生成最终回答。
```

## 6. 代码结构

```text
app/
  main.py                    FastAPI 入口
  config.py                  .env 配置读取
  schemas.py                 API 与内部数据结构
  session_store.py           MySQL 会话存储
  domain/
    constants.py             体质、地区、节气、建议类型等领域常量
    normalizers.py           地区/体质/节气到季节/建议类型/症状归一化
  nlp/
    intent_parser.py         意图识别和槽位抽取
    clarification.py         追问决策
  rag/
    chunk_builder.py         Excel/TXT 分块
    qdrant_store.py          Qdrant 与 embedding 封装
    constitution_identifier.py
    retriever.py
    answer_generator.py
  services/
    chat_service.py          问答主链路编排
scripts/
  build_index.py             构建并写入两个 Qdrant collection
```
