import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from src.config import PROJECT_ROOT
from src.rag import rag
from src.retrieval import get_greeting_response
from src.session import get_session_workflow
from src.speech2text import speech2text
from src.text2speech import text2speech
from utils.autoplay_audio import autoplay_audio
from utils.clean_for_tts import clean_for_tts
from utils.remove_emojis import remove_emojis

logger = logging.getLogger(__name__)


st.image(str(PROJECT_ROOT / "header.jpeg"), use_container_width=True)
st.title("Fabrizio Romano Q&A AI Chatbot")
st.sidebar.subheader("Select an LLM model")
llm_choice = st.sidebar.selectbox("Choose LLM Model", ["Groq", "OpenAI"])
api_key = st.sidebar.text_input(
    f"Enter {llm_choice} API Key:", type="password", key=f"{llm_choice}_api_key"
)
voice_available = bool(os.getenv("DG_API_KEY"))
read_aloud = st.sidebar.checkbox("Read answers aloud", value=False, disabled=not voice_available)
if not voice_available:
    st.sidebar.caption("Voice input and playback require DG_API_KEY. Text chat is available.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am a Fabrizio Romano AI bot. Ask me any transfer/football questions Fabrizio has tweeted about in the past."),
    ]
if "audio_bytes" not in st.session_state:
    st.session_state.audio_bytes = None

for message in st.session_state.chat_history:
    with st.chat_message("assistant" if isinstance(message, AIMessage) else "user"):
        st.markdown(message.content)

user_query = st.chat_input("Ask a question...")
audio = st.audio_input("Record a voice query/question", disabled=not voice_available)
audio_bytes = audio.getvalue() if audio else None
is_new_audio = audio_bytes is not None and audio_bytes != st.session_state.audio_bytes

if user_query or is_new_audio:
    with TemporaryDirectory(prefix="fabrizio-chat-") as audio_dir:
        if is_new_audio:
            st.session_state.audio_bytes = audio_bytes
            audio_query = Path(audio_dir) / "query.wav"
            audio_query.write_bytes(audio_bytes)
            try:
                with st.spinner("Transcribing audio..."):
                    user_query = speech2text(str(audio_query))
                    if not user_query or not user_query.strip():
                        raise ValueError("No speech was detected. Please try again.")
            except Exception as exc:
                logger.exception("Audio transcription failed")
                st.error(f"Failed to transcribe audio: {exc}")
                st.stop()

        st.session_state.chat_history.append(HumanMessage(content=user_query))
        with st.chat_message("user"):
            if is_new_audio:
                st.audio(audio_bytes, format="audio/wav")
            st.markdown(user_query)

        with st.chat_message("assistant"):
            try:
                full_response = get_greeting_response(user_query)
                if full_response is None:
                    workflow = get_session_workflow(st.session_state, llm_choice, api_key)
                    with st.spinner("Searching saved tweets and generating a response..."):
                        full_response, chunks, urls = rag(
                            st.session_state.rag_thread_id, workflow, user_query
                        )
                st.markdown(full_response)
                st.session_state.chat_history.append(AIMessage(content=full_response))
            except Exception as exc:
                logger.exception("Chat response failed")
                st.error(f"Unable to answer your question: {exc}")
                st.stop()

            if read_aloud:
                try:
                    with st.spinner("Generating audio..."):
                        audio_file = text2speech(
                            clean_for_tts(remove_emojis(full_response)),
                            filename=str(Path(audio_dir) / "response.wav"),
                        )
                    if not audio_file:
                        raise ValueError("The speech service returned no audio.")
                    autoplay_audio(audio_file)
                except Exception:
                    logger.exception("Audio playback failed")
                    st.warning("Your answer is ready, but audio playback is unavailable.")
