import os
from dotenv import load_dotenv
from groq import Groq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

load_dotenv()

groq_api = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    api_key=groq_api,
    model="llama3-8b-8192",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    stream=True,
)

template = """
You're a helpful football journalist assistant. You impersonate a popular football journalist, Fabrizio Romano.
Answer the QUESTION based on the CONTEXT from the database of his scraped tweets.
Use only the facts from the CONTEXT when answering the QUESTION.

<context>
{context}
</context>

QUESTION: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | llm | StrOutputParser()

def ask_llm(query, context):    
     return chain.stream({
        "context": context,
        "question": query,
    })
