from functools import lru_cache

import lancedb
import pandas as pd
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector

from src.config import PROJECT_ROOT, get_database_settings


@lru_cache(maxsize=1)
def get_tweet_schema():
    model = get_registry().get("sentence-transformers").create(name="all-mpnet-base-v2")

    class TweetDocument(LanceModel):
        tweet_count: int
        tweet_id: int
        username: str
        text: str = model.SourceField()
        created_at: str
        url: str
        vector: Vector(model.ndims()) = model.VectorField()

    return TweetDocument


def initialize_database():
    db_path, table_name = get_database_settings()
    db = lancedb.connect(db_path)
    if table_name in db.table_names():
        return db.open_table(table_name)

    tweets_df = pd.read_csv(PROJECT_ROOT / "data" / "fabrizioromano_tweets.csv")
    data = tweets_df.rename(columns={"created at": "created_at"}).to_dict("records")
    table = db.create_table(table_name, data=data, schema=get_tweet_schema())
    table.create_fts_index("text")
    return table
