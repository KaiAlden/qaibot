from __future__ import annotations

import argparse
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
