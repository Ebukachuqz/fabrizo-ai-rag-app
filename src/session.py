"""Keep the active model and conversation checkpoints in one Streamlit session."""

from hashlib import sha256

from langgraph.checkpoint.memory import MemorySaver

from src.langgraph_workflow import initialize_rag_workflow
from src.prompt import get_api_key, get_model_name, initialize_llm


def get_session_workflow(state, choice, api_key=None):
    resolved_key = get_api_key(choice, api_key)
    signature = (choice, get_model_name(choice), sha256(resolved_key.encode()).hexdigest())
    if state.get("rag_model_signature") != signature or "langgraph_workflow" not in state:
        llm = initialize_llm(choice, resolved_key)
        memory = state.get("rag_memory")
        if memory is None:
            memory = MemorySaver()
        state["langgraph_workflow"] = initialize_rag_workflow(llm, memory=memory)
        state["rag_memory"] = memory
        state["rag_model_signature"] = signature
        state.setdefault("rag_thread_id", "1")
    return state["langgraph_workflow"]
