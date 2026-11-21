from src.retrieval import search_tweets
from src.prompt import ask_llm


def rag(thread_id, workflow, query):
    """Return an answer, display chunks and sources; propagate failures to the UI."""
    retrieved_context, urls = search_tweets(query)
    if not retrieved_context:
        response = "I couldn't find matching tweets. Try including a player or club name."
        return response, list(response), []
    context = "\n".join(retrieved_context)
    full_response, chunks = ask_llm(thread_id, workflow, query, context)
    return full_response, chunks, urls
