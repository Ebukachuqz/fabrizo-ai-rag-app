import os
import logging
from deepgram.utils import verboselogs
from dotenv import load_dotenv

from deepgram import (
    DeepgramClient,
    SpeakOptions,
)

load_dotenv()
api_key = os.getenv("DG_API_KEY")



def text2speech(text, filename):
    try:
        SPEAK_OPTIONS = {"text": text}
        deepgram = DeepgramClient(api_key=api_key)

        options = SpeakOptions(
            model="aura-helios-en",
            encoding="linear16",
        )

        response = deepgram.speak.rest.v("1").save(filename, SPEAK_OPTIONS, options)
        # print(response.to_json(indent=4))
        return filename

    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    text2speech("Read this [Tweet](www.example.com)")