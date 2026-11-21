from src.vectordb import lancedb_hybrid_search
from src.prompt import ask_llm


def rag(thread_id, workflow, query):
    """Return an answer, display chunks and sources; propagate failures to the UI."""
    retrieved_context, urls = lancedb_hybrid_search(query)
    context = "\n".join(retrieved_context)
    full_response, chunks = ask_llm(thread_id, workflow, query, context)
    return full_response, chunks, urls
