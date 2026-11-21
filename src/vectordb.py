from functools import lru_cache

import lancedb
from lancedb.rerankers import CrossEncoderReranker

from src.config import get_database_settings


@lru_cache(maxsize=1)
def get_reranker():
    # Download/load the model only when a query actually needs retrieval.
    return CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-2-v2")


def lancedb_hybrid_search(query, k=3):
    db_path, table_name = get_database_settings()
    db = lancedb.connect(db_path)
    table = db.open_table(table_name)
    results = (
        table.search(query, query_type="hybrid")
        .rerank(reranker=get_reranker())
        .limit(k)
        .to_list()
    )
    context = [
        f"On {result['created_at']}, Fabrizio Romano tweeted: {result['text']}, "
        f"Tweet URL: {result['url']}"
        for result in results
    ]
    return context, [result["url"] for result in results]
