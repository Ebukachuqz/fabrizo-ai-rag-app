from src.vectordb import lancedb_hybrid_search
from src.prompt import ask_llm

def rag(query):
    retrieved_context, urls = lancedb_hybrid_search(query)
    context = "\n".join(retrieved_context)
    response_stream = ask_llm(query, context)
    return response_stream, urls
