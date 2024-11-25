import os
from dotenv import load_dotenv
import lancedb
from lancedb.rerankers import LinearCombinationReranker, CrossEncoderReranker

load_dotenv()

db_path = os.getenv("LANCEDB_PATH")
table_name = os.getenv("LANCEDB_TABLE")

cross_encoder_model = "cross-encoder/ms-marco-MiniLM-L-2-v2"
ce_reranker = CrossEncoderReranker(model_name=cross_encoder_model)
lc_reranker = LinearCombinationReranker(weight=0.7)

def lancedb_hybrid_search(query, k=5):
    db = lancedb.connect(db_path)
    table = db.open_table(table_name)
    
    results = table.search(query, query_type="hybrid").rerank(reranker=ce_reranker).limit(k).to_list()

    # Format context to include both text and created_at date
    context = [f"On {result['created_at']}, Fabrizio Romano tweeted: {result['text']}, Tweet URL: {result['url']}" for result in results]
    urls = [result['url'] for result in results]
    
    return context, urls