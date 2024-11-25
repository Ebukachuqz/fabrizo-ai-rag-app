import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from init_lancedb import initialize_database
from src.rag import rag
from src.feedback import store_feedback
from src.text2speech import text2speech
from src.speech2text import speech2text
from utils.remove_emojis import remove_emojis
import time
import re
from utils.autoplay_audio import autoplay_audio
from utils.get_ratings_from_emoji import get_rating_from_emoji
import os
from streamlit_feedback import streamlit_feedback
from src.prompt import initialize_llm
from src.langgraph_workflow import initialize_rag_workflow
from utils.clean_for_tts import clean_for_tts

def feedback_cb():
    feedback = st.session_state.fb_k
    if feedback:
        rating = get_rating_from_emoji(feedback['score'])
        latest_human_query = st.session_state.chat_history[-2].content
        latest_ai_response = st.session_state.chat_history[-1].content

        # Store feedback status with the corresponding AI response
        st.session_state.feedback_status[latest_ai_response] = {
            'rating': rating,
            'emoji': feedback['score']
        }
        
        try:
            store_feedback(latest_human_query, latest_ai_response, rating)
            st.toast("Thank you for your feedback!", icon="✨")
        except Exception as e:
            st.toast("Error saving feedback", icon="🚨")
            print(f"Error storing feedback: {str(e)}")

audio_response = "audio_response.mp3"
audio_query = "audio_query.wav"

st.image("header.jpeg", use_container_width=True)
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
        AIMessage(content="Hello, I am a Fabrizo Romano AI bot. Ask me any transfer/football questions Fabrizo has tweeted about in the past."),
    ]

if "audio_bytes" not in st.session_state:
    st.session_state.audio_bytes = None

if "feedback_status" not in st.session_state:
    st.session_state.feedback_status = {}

llm = initialize_llm(llm_choice, api_key)

# Initialize session state for langgraph workflow
if 'langgraph_workflow' not in st.session_state:
    st.session_state.langgraph_workflow = initialize_rag_workflow(llm)
    st.session_state.rag_thread_id = "1"

# Display conversation history
for message in st.session_state.chat_history:
    with st.chat_message("AI" if isinstance(message, AIMessage) else "Human"):
        st.write(message.content)

        if isinstance(message, AIMessage):
            feedback = st.session_state.feedback_status.get(message.content)
            if feedback:
                st.markdown(f"**_Rating:_** {feedback['emoji']}")

user_query = st.chat_input("Ask a question...")
audio = st.audio_input("Record a voice query/question")

def check_new_audio():
    global audio

    if not audio:
        return False

    if audio and audio.getvalue() == st.session_state.audio_bytes:
        return False
    else:
        st.session_state.audio_bytes = audio.getvalue()
        return True

# Process user input
is_new_audio = check_new_audio()
if user_query or is_new_audio:
    if os.path.exists(audio_response):
        os.remove(audio_response)
    if is_new_audio:
        with open(audio_query, "wb") as f:
            f.write(audio.read())
        with st.chat_message("Human"):
            try:
                with st.spinner("Transcribing audio..."):
                    transcribed_text = speech2text(audio_query)
                    if not transcribed_text or transcribed_text.strip() == "":
                        raise ValueError("No text transcribed")
                    
                    st.session_state.chat_history.append(HumanMessage(content=transcribed_text))
                    autoplay_audio(audio_query)
                    st.markdown(transcribed_text)
                    user_query = transcribed_text
            except Exception as e:
                st.error(f"Failed to transcribe audio. Error: {e}")
                st.stop()
            finally:
                os.remove(audio_query)
                    
    elif user_query:
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        with st.chat_message("Human"):
            st.markdown(user_query)
    else:
        st.stop()

    with st.chat_message("AI"):
        response_container = st.empty()
        response_text = ""

        try:
            with st.spinner("Generating response..."):
                full_response, chunks, urls = rag(st.session_state.rag_thread_id, st.session_state.langgraph_workflow, user_query, llm_choice, api_key)

            with st.spinner("Generating audio..."):
                audio_file = text2speech(clean_for_tts(remove_emojis(full_response)), filename=audio_response)
            if audio_file:
                autoplay_audio(audio_response)

            for chunk in chunks:
                response_text += chunk
                response_container.write(response_text)
                time.sleep(0.03)

            st.session_state.chat_history.append(AIMessage(content=response_text))

            with st.form('form'):
                streamlit_feedback(feedback_type="faces", align="flex-start", key='fb_k')
                st.form_submit_button('Submit feedback', on_click=feedback_cb)

        except Exception as e:
            error_str = str(e)
            match = re.search(r"'message':\s'(.+?)'", error_str)
            if match:
                error_message = match.group(1)
                st.error(error_message)
            else:
                st.error(str(e))