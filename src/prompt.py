import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables from the .env file
load_dotenv()

groq_api = os.getenv("GROQ_API_KEY")

# Define the query prompt template
query_prompt_template = """
You're a helpful football journalist assistant. You impersonate a popular football journalist, Fabrizio Romano.
Answer the QUESTION based on the CONTEXT from the database of his scraped tweets.
Use only the facts from the CONTEXT when answering the QUESTION.

<context>
{context}
</context>

QUESTION: {question}
""".strip()

# Initialize Groq client
client = Groq(
    api_key=groq_api,
)

# Function to construct the prompt
def construct_prompt(query, context):
    return query_prompt_template.format(question=query, context=context).strip()

# Function to ask the Groq LLM
def ask_groq_llm(prompt):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama3-8b-8192",
    )
    
    return chat_completion.choices[0].message.content
