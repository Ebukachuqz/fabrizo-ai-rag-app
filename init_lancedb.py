import os
from dotenv import load_dotenv
load_dotenv()

db_path = os.getenv("LANCEDB_PATH")
table_name = os.getenv("LANCEDB_TABLE")

import pandas as pd
import lancedb
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import get_registry

db = lancedb.connect(db_path)
model = get_registry().get("sentence-transformers").create(name="all-mpnet-base-v2")

# Data injestion
tweets_df = pd.read_csv("data/fabrizioromano_tweets.csv")

class TweetDocument(LanceModel):
    tweet_count: int
    tweet_id: int
    username: str
    text: str = model.SourceField()
    created_at: str
    url: str
    vector: Vector(model.ndims()) = model.VectorField()

data = tweets_df.apply(
    lambda row: {
        "tweet_count": row["tweet_count"],
        "tweet_id": row["tweet_id"],
        "username": row["username"],
        "text": row["text"],
        "created_at": row["created at"],
        "url": row["url"]
    },
    axis=1
).values.tolist()

table = db.create_table(table_name, schema=TweetDocument)

table.add(data)

table.create_fts_index("text")
print("Table created and data added")