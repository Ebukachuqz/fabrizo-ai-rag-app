import streamlit as st
from init_lancedb import initialize_database
from src.rag import rag
from src.feedback import store_feedback

st.title("Twitter Q&A with RAG (Groq)")

with st.spinner("Setting up the database..."):
    initialize_database()

user_query = st.text_input("Ask a question:")

if "answer" not in st.session_state:
    st.session_state.answer = None
    st.session_state.references = []

if st.button("Get Answer"):
    st.session_state.answer, st.session_state.references = rag(user_query)

# Display the LLM answer if it exists
if st.session_state.answer:
    st.write("### Answer:")
    st.write(st.session_state.answer)

    if st.session_state.references:
        st.write("### References:")
        for url in st.session_state.references[:3]:  
            st.markdown(f"- [Tweet]({url})")
else:
    st.write("Ask a question to get an answer.")

# Only show feedback options if an answer exists
if st.session_state.answer:
    # Collect user feedback on the response
    rating = st.slider("Rate the response (1-5):", 1, 5)

    if st.button("Submit Feedback"):
        try:
            # Use session state to get the answer and query
            store_feedback(user_query, st.session_state.answer, rating)
            st.write("Thank you for your feedback!")
        except Exception as e:
            st.error(f"Error submitting feedback: {str(e)}")
