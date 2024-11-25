from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, RemoveMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, StateGraph, START, END
from typing import Literal

class State(MessagesState):
    """Enhanced state class with summary and context management"""
    summary: str
    context: str

def initialize_rag_workflow(llm):
    """Create a RAG workflow with LangGraph memory management"""
    memory = MemorySaver()
    workflow = StateGraph(State)
    
    def call_model(state: State):
        summary = state.get("summary", "")
        context = state.get("context", "")
        
        system_prompt = f"""
        You are Fabrizio Romano, a renowned football journalist known for your accurate transfer news and signature catchphrase "Here we go!".

        {f'Previous conversation summary: {summary}' if summary else ''}

        Current Context:
        {context}

        Instructions:
        - Answer strictly based on the facts in the provided Context.
        - Use my typical reporting style - clear, concise, and confident.
        - For transfer news, specify the stage of negotiations, clubs involved, and player status.
        - Use "Here we go!" only when a deal is confirmed.
        - If the context doesn't support a confident answer, say "I have no updates on this situation at the moment."
        - Avoid speculation or inferring details not present in the context.
        - Keep responses focused and newsy, matching my Twitter style.
        """
        
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        response = llm.invoke(messages)
        return {"messages": [response]}
        
    def should_continue(state: State) -> Literal["summarize_conversation", END]:
        """Determine if conversation should continue or be summarized"""
        messages = state["messages"]
        if len(messages) > 6:  # Summarize after 6 messages
            return "summarize_conversation"
        return END

    def summarize_conversation(state: State):
        """Generate conversation summary"""
        summary = state.get("summary", "")
        if summary:
            summary_prompt = (
                f"Previous summary: {summary}\n\n"
                "Extend the summary by taking into account relvant details of the conversation from the new messages above:"
            )
        else:
            summary_prompt = (
                "Distill the above chat messages into a single summary message. "
                "Include as many specific details as you can:"
            )

        messages = state["messages"] + [HumanMessage(content=summary_prompt)]
        response = llm.invoke(messages)
        # Keep last two messages for context
        delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
        return {
            "summary": response.content,
            "messages": delete_messages
        }

    # Define nodes and edges
    workflow.add_node("conversation", call_model)
    workflow.add_node("summarize_conversation", summarize_conversation)
    # Set entry point
    workflow.add_edge(START, "conversation")
    # Add conditional edge for summarization
    workflow.add_conditional_edges(
        "conversation",
        should_continue,
    )
    # Add edge from summarization to end
    workflow.add_edge("summarize_conversation", END)
    return workflow.compile(checkpointer=memory)

