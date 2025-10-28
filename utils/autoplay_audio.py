import base64

def autoplay_audio(audio_file):
    """Render an autoplaying audio element when running under Streamlit.

    Safely no-ops when Streamlit isn't initialized (e.g., running as a script/tests).
    """
    with open(audio_file, "rb") as audio_file:
        audio_bytes = audio_file.read()
    base64_audio = base64.b64encode(audio_bytes).decode("utf-8")
    audio_html = f'<audio src="data:audio/mp3;base64,{base64_audio}" controls autoplay>'

    # Lazy-import Streamlit and guard on active context to avoid SessionInfo errors
    try:
        import streamlit as st  # imported here to avoid module import-time side effects
        try:
            # If there's no active Streamlit script context, just return silently
            from streamlit.runtime.scriptrunner import get_script_run_ctx
            if get_script_run_ctx() is None:
                return
        except Exception:
            # Older/newer Streamlit without scriptrunner API; fall back to best-effort
            pass

        st.markdown(audio_html, unsafe_allow_html=True)
    except Exception:
        # Streamlit not installed or runtime not initialized; safely skip rendering
        return