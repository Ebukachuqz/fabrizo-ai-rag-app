import os
from dotenv import load_dotenv
from groq import Groq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

load_dotenv()

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
    stream=True,
)

# Define the prompt template
template = """
You're a helpful football journalist assistant. You impersonate a popular football journalist, Fabrizio Romano.
Answer the QUESTION based on the CONTEXT from the database of his scraped tweets.
Use only the facts from the CONTEXT when answering the QUESTION.
Do not make up responses that are not part of the context. Say "I don't know" or "I don't have sufficient data on that" if you can't find the answer in the context.

<context>
{context}
</context>

QUESTION: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

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
        # Default to Groq with default API key if no valid choice is provided
        return default_llm

def ask_llm(query, context, llm_choice="Groq", api_key=None):
    # Initialize LLM based on user choice or default
    llm = initialize_llm(llm_choice, api_key)
    chain = prompt | llm | StrOutputParser()

    try:
        return chain.stream({
            "context": context,
            "question": query,
        })
    except Exception as e:
        raise ValueError(f"Error generating response: {str(e)}")
