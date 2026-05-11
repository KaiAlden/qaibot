# 项目启动前手动执行一次脚本，将原始数据文件中的内容构建成知识片段，并存入 Qdrant 向量数据库中。
from __future__ import annotations

import argparse # 命令行参数解析工具，用于从命令行获取输入参数，如数据目录路径和是否进行干运行（dry run）等。
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import load_settings
from app.rag.chunk_builder import build_all_chunks
from app.rag.qdrant_store import QdrantStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Build QAibot TCM RAG index in Qdrant.")
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--dry-run", action="store_true", help="Only build chunks and print summary.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    chunks = build_all_chunks(data_dir)
    by_type: dict[str, int] = {}
    for chunk in chunks:
        by_type[chunk.type] = by_type.get(chunk.type, 0) + 1

    print(f"Built {len(chunks)} chunks from {data_dir}")
    for chunk_type, count in sorted(by_type.items()):
        print(f"- {chunk_type}: {count}")

    if args.dry_run:
        return

    constitution_chunks = [chunk for chunk in chunks if chunk.type == "constitution_identify"]
    advice_chunks = [chunk for chunk in chunks if chunk.type in {"diet_principle", "suggestion"}]

    settings = load_settings()
    store = QdrantStore(settings)
    constitution_total = store.upsert_chunks(settings.qdrant_constitution_collection, constitution_chunks)
    advice_total = store.upsert_chunks(settings.qdrant_advice_collection, advice_chunks)

    print(
        f"Upserted {constitution_total} chunks into "
        f"Qdrant collection {settings.qdrant_constitution_collection}"
    )
    print(
        f"Upserted {advice_total} chunks into "
        f"Qdrant collection {settings.qdrant_advice_collection}"
    )


if __name__ == "__main__":
    main()



"""
原始数据文件（Excel/CSV等）
         │
         ▼
  build_all_chunks(data_dir)     ← 将数据拆分成知识片段
         │
         ├── constitution_identify 类型（体质辨识资料）
         ├── diet_principle 类型    （季节饮食原则）
         └── suggestion 类型        （调理建议）
         │
         ▼
  按类型分组
         │
         ├── constitution_chunks → Qdrant 集合1（体质库）
         │
         └── advice_chunks       → Qdrant 集合2（建议库）
                │
                ▼
         store.upsert_chunks(集合名, chunks)  ← 存入 Qdrant


### 1. 数据分片（`build_all_chunks`）

读取 `data/` 目录下的原始数据（可能是 Excel 文件），将每行/每条记录转换为 `KnowledgeChunk` 对象。

### 2. 向量化并存入 Qdrant（`upsert_chunks`）

每个 `KnowledgeChunk` 在存入 Qdrant 时会：
1. 通过 embedding 模型转为向量
2. 附带 payload 元数据（chunk_id、type、content、area、season 等）
3. 存入对应的集合

"""