import os
import requests

API_KEY = os.environ.get("GROQ_API_KEY")

def transcribe_audio(audio_path):
    url = "https://api.groq.com/openai/v1/audio/transcriptions"

    headers = {
        "Authorization": f"Bearer {API_KEY}"
    }

    with open(audio_path, "rb") as audio:
        files = {
            "file": (os.path.basename(audio_path), audio)
        }

        data = {
            "model": "whisper-large-v3-turbo",
            "response_format": "json"
        }

        response = requests.post(
            url,
            headers=headers,
            files=files,
            data=data,
            timeout=120
        )

    response.raise_for_status()
    return response.json()["text"]


if __name__ == "__main__":
    audio_path = input("Audio file: ").strip()

    try:
        text = transcribe_audio(audio_path)
        print("\nTranscript:")
        print(text)
    except requests.RequestException as e:
        print(f"API error: {e}")
