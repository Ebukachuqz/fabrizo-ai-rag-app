from src.vectordb import lancedb_hybrid_search
from src.prompt import ask_llm

def rag(thread_id_state, langgraph_workflow_state, query, llm_choice, api_key):
    try:
        retrieved_context, urls = lancedb_hybrid_search(query)
        context = "\n".join(retrieved_context)
        full_response, chunks = ask_llm(thread_id_state, langgraph_workflow_state, query, context, llm_choice, api_key)
        return full_response, chunks, urls

    except Exception as e:
        # Return error message and no URLs to indicate failure
        error_message = f"An error occurred during the RAG process: {e}"
        return error_message, None
