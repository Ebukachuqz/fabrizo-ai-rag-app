from src.vectordb import lancedb_hybrid_search
from src.prompt import ask_llm

def rag(query, llm_choice, api_key):
    try:
        retrieved_context, urls = lancedb_hybrid_search(query)
        context = "\n".join(retrieved_context)
        response = ask_llm(query, context, llm_choice, api_key)
        return response, urls

    except Exception as e:
        # Return error message and no URLs to indicate failure
        error_message = f"An error occurred during the RAG process: {e}"
        return error_message, None
