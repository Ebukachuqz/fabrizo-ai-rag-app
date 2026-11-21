import os
from src import config

from deepgram import (
    DeepgramClient,
    SpeakOptions,
)




def text2speech(text, filename):
    try:
        api_key = os.getenv("DG_API_KEY")
        if not api_key:
            raise ValueError("Audio playback requires DG_API_KEY.")
        SPEAK_OPTIONS = {"text": text}
        deepgram = DeepgramClient(api_key=api_key)

        options = SpeakOptions(
            model="aura-helios-en",
            encoding="linear16",
        )

        deepgram.speak.rest.v("1").save(filename, SPEAK_OPTIONS, options)
        return filename

    except Exception as e:
        raise RuntimeError(f"Speech generation failed: {e}") from e

if __name__ == "__main__":
    text2speech("Here we go!", "response.wav")
