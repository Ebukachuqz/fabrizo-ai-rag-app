import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from init_lancedb import initialize_database
from src.rag import rag
from src.feedback import store_feedback
import time
import re

st.image("header.jpeg", use_column_width=True)

st.title("Fabrizio Romano Q&A AI Chatbot")

# Set up LLM choice
llm_options = ["Groq", "OpenAI"]
st.sidebar.subheader("Select an LLM model")
st.sidebar.caption("Default is Groq. Easy on my credits tho😭")
st.sidebar.text("If you want to try other LLM models.")
llm_choice = st.sidebar.selectbox("Choose LLM Model", llm_options)
api_key = st.sidebar.text_input(f"Enter {llm_choice} API Key:", type="password")

with st.spinner("Setting up the database. This may take 3-6 minutes..."):
    initialize_database()

# Initialize chat history and feedback state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am a Fabrizo Romano AI bot. Ask me any transfer/football questions Fabrizo has tweeted about. From Jul 25 2024 to Sep 07 2024"),
    ]

if "feedback_ready" not in st.session_state:
    st.session_state.feedback_ready = False

# Display conversation history
for message in st.session_state.chat_history:
    with st.chat_message("AI" if isinstance(message, AIMessage) else "Human"):
        st.write(message.content)

user_query = st.chat_input("Ask a question...")

# Process user query
if user_query:
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        response_container = st.empty()
        response_text = ""

        try:
            with st.spinner("Generating response..."):
                response_stream, urls = rag(user_query, llm_choice, api_key)
                
            # Handle response or display error notification
            if urls is None:  # An error occurred during RAG
                st.error(response_stream)
            else:
                for chunk in response_stream:
                    response_text += chunk
                    response_container.write(response_text)
                    time.sleep(0.03)

                # Display references if any
                if urls:
                    references_text = "\n\n##### References:\n" + "\n".join(f"- [Tweet]({url})" for url in urls[:3])
                    response_text += references_text  
                    response_container.write(response_text) 

            st.session_state.chat_history.append(AIMessage(content=response_text))
            st.session_state.feedback_ready = True

        except Exception as e:
            error_str = str(e)
            match = re.search(r"'message':\s'(.+?)'", error_str)
            if match:
                error_message = match.group(1)
                st.error(error_message)
            else:
                st.error(str(e))

# Feedback section
if st.session_state.feedback_ready:
    rating = st.slider("Rate the response (1-5):", 1, 5)
    if st.button("Submit Feedback"):
        with st.spinner("Submitting feedback..."):
            try:
                latest_response = st.session_state.chat_history[-1].content
                store_feedback(user_query, latest_response, rating)
                st.write("Thank you for your feedback!")
                st.session_state.feedback_ready = False
            except Exception as e:
                st.error(f"Error submitting feedback: {str(e)}")
