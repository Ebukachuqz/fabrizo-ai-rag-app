import os

from langchain_core.messages import AIMessage, HumanMessage
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from src import config  # Load the project's .env before resolving credentials.


def get_api_key(choice, api_key=None):
    if choice not in ("Groq", "OpenAI"):
        raise ValueError(f"Unsupported LLM provider: {choice}")
    variable = "GROQ_API_KEY" if choice == "Groq" else "OPENAI_API_KEY"
    resolved_key = (api_key or "").strip() or os.getenv(variable, "").strip()
    if not resolved_key:
        raise ValueError(f"Enter a {choice} API key in the sidebar or configure {variable}.")
    return resolved_key


def get_model_name(choice):
    if choice == "Groq":
        return os.getenv("GROQ_MODEL") or "qwen/qwen3.8-27b"
    return os.getenv("OPENAI_MODEL") or "gpt-3.5-turbo"


def initialize_llm(choice, api_key=None):
    """Create only the selected provider, using its own credentials."""
    resolved_key = get_api_key(choice, api_key)
    if choice == "Groq":
        return ChatGroq(
            api_key=resolved_key,
            model=get_model_name(choice),
            temperature=0,
            timeout=60,
            max_retries=2,
            streaming=False,
        )
    return ChatOpenAI(
        api_key=resolved_key,
        model=get_model_name(choice),
        temperature=0,
        timeout=60,
        max_retries=2,
        streaming=False,
    )


def ask_llm(thread_id, workflow, query, context):
    """Invoke the session's configured workflow without constructing another LLM."""
    final_state = workflow.invoke(
        {"messages": [HumanMessage(content=query)], "context": context},
        {"configurable": {"thread_id": thread_id}},
    )
    ai_message = next(
        (msg for msg in reversed(final_state["messages"]) if isinstance(msg, AIMessage)),
        None,
    )
    if ai_message is None:
        raise ValueError("The model returned no answer. Please try again.")
    full_response = ai_message.content
    return full_response, list(full_response)
