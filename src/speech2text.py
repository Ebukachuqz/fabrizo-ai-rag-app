import os
from src import config

from deepgram import (
    DeepgramClient,
    PrerecordedOptions,
)

def speech2text(audio):
    try:
        api_key = os.getenv("DG_API_KEY")
        if not api_key:
            raise ValueError("Voice input requires DG_API_KEY.")
        deepgram = DeepgramClient(api_key=api_key)

        with open(audio, "rb") as file:
            buffer_data = file.read()

        payload = {
            "buffer": buffer_data,
        }

        options = PrerecordedOptions(
            model="nova-2",
            smart_format=True,
        )

        response = deepgram.listen.rest.v("1").transcribe_file(payload, options)

        return response.results.channels[0].alternatives[0].transcript

    except Exception as e:
        raise RuntimeError(f"Speech transcription failed: {e}") from e


if __name__ == "__main__":
    speech2text("test.mp3")
