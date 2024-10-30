import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from init_lancedb import initialize_database
from src.rag import rag
from src.feedback import store_feedback
import time

st.title("Fabrizo Romano Q&A Chatbot")

with st.spinner("Setting up the database..."):
    initialize_database()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am a Fabrizo Romano AI bot. Ask me any transfer/football questions Fabrizo has tweeted about"),
    ]

if "feedback_ready" not in st.session_state:
    st.session_state.feedback_ready = False
    
# conversation
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

user_query = st.chat_input("Ask a question...")
if user_query:
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        response_container = st.empty()
        response_text = ""

        with st.spinner("Generating response..."):
            response_stream, urls = rag(user_query)
        for chunk in response_stream:
            response_text += chunk
            response_container.write(response_text)
            time.sleep(0.03)

        if urls:
            references_text = "\n\n##### References:\n" + "\n".join(f"- [Tweet]({url})" for url in urls[:3])
            response_text += references_text  
            response_container.write(response_text) 

    st.session_state.chat_history.append(AIMessage(content=response_text))
    st.session_state.feedback_ready = True

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
