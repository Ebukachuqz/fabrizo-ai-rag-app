from src.vectordb import lancedb_hybrid_search
from src.prompt import construct_prompt, ask_groq_llm

def rag(query):
    retrieved_context, urls = lancedb_hybrid_search(query)
    context = "\n".join(retrieved_context)
    prompt = construct_prompt(query, context)
    response = ask_groq_llm(prompt)
    return response, urls
