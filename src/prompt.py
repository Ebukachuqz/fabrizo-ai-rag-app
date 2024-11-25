import openlit
import os
from dotenv import load_dotenv
from groq import Groq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, RemoveMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, StateGraph, START, END
import streamlit as st
from typing import Literal, Dict, Any, Optional, Tuple, List

openlit.init()
load_dotenv()

class State(MessagesState):
    """Enhanced state class with summary and context management"""
    summary: str
    context: str

# Load default Groq API key
default_groq_api = os.getenv("GROQ_API_KEY")

# Default Groq LLM setup
default_llm = ChatGroq(
    api_key=default_groq_api,
    model="llama3-8b-8192",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    stream=False,
)

def initialize_llm(choice, api_key=None):
    """Initialize the LLM based on user choice."""
    if choice == "Groq" and api_key:
        return ChatGroq(
            api_key=api_key,
            model="llama3-8b-8192",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            stream=True,
        )
    elif choice == "OpenAI" and api_key:
        return ChatOpenAI(
            api_key=api_key,
            model="gpt-3.5-turbo",
            temperature=0,
            max_tokens=None,
            stream=True,
        )
    else:
        return default_llm

def ask_llm(thread_id_state, langgraph_workflow_state, query, context, llm_choice="Groq", api_key=None):
    """Main function to process queries with LangGraph memory"""
    try:
        # Initialize LLM
        current_llm = initialize_llm(llm_choice, api_key)
        
        # Create new workflow if none exists
        # workflow = create_rag_workflow(default_llm)
        
        # Create input state
        input_state = {
            "messages": [HumanMessage(content=query)],
            "context": context,
        }
        
        # Process through workflow
        config = {"configurable": {"thread_id": thread_id_state}}
        
        # Execute workflow and get final state
        final_state = langgraph_workflow_state.invoke(input_state, config)
        
        # Extract the last AI message
        messages = final_state["messages"]
        ai_message = next(msg for msg in reversed(messages) 
                         if isinstance(msg, AIMessage))
        
        full_response = ai_message.content
        # Create chunks for streaming
        chunks = [char for char in full_response]
        
        return full_response, chunks
        
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        raise ValueError(f"Error generating response: {str(e)}")