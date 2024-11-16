import os
import logging
from deepgram.utils import verboselogs
from dotenv import load_dotenv

from deepgram import (
    DeepgramClient,
    PrerecordedOptions,
)
load_dotenv()
api_key = os.getenv("DG_API_KEY")

def speech2text(audio):
    try:
        deepgram: DeepgramClient = DeepgramClient(api_key=api_key)

        with open(audio, "rb") as file:
            buffer_data = file.read()

        payload: FileSource = {
            "buffer": buffer_data,
        }

        options = PrerecordedOptions(
            model="nova-2",
            smart_format=True,
        )

        response = deepgram.listen.rest.v("1").transcribe_file(payload, options)

        transcript = response['results']['channels'][0]['alternatives'][0]['transcript']

        print(f"Transcript: {transcript}")
        return transcript

    except Exception as e:
        print(f"Exception: {e}")


if __name__ == "__main__":
    speech2text("test.mp3")