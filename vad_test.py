

import webrtcvad
import wave
import subprocess
from pathlib import Path

RECORDINGS_DIR = Path("recordings")
SUPPORTED_FORMATS = {".m4a", ".wav", ".mp3", ".flac", ".ogg", ".aac"}


def find_recordings():
    return sorted(
        [
            file for file in RECORDINGS_DIR.iterdir()
            if file.is_file()
            and file.suffix.lower() in SUPPORTED_FORMATS
            and not file.name.startswith("_temp_")
        ],
        key=lambda file: file.name.lower()
    )


def select_recording(files):
    if not files:
        print("No recordings found.")
        return None

    print("\nAvailable recordings:")
    for i, file in enumerate(files, start=1):
        print(f"{i}. {file.name}")

    while True:
        choice = input("\nChoose a recording (or q to quit): ").strip()

        if choice.lower() == "q":
            return None

        if choice.isdigit() and 1 <= int(choice) <= len(files):
            return files[int(choice) - 1]

        print("Invalid choice.")


def convert_to_wav(source):
    output = RECORDINGS_DIR / "_temp_vad.wav"

    command = [
        "ffmpeg", "-y",
        "-hide_banner", "-loglevel", "error",
        "-i", str(source),
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(output)
    ]

    subprocess.run(command, check=True)
    return output


def analyze_vad(wav_file):
    vad = webrtcvad.Vad(2)

    frame_ms = 30
    speech_frames = 0
    total_frames = 0
    frame_bytes = 960  # 30 ms at 16 kHz, 16-bit mono

    with wave.open(str(wav_file), "rb") as audio:
        if audio.getframerate() != 16000:
            raise ValueError("Audio must be 16 kHz.")

        if audio.getnchannels() != 1 or audio.getsampwidth() != 2:
            raise ValueError("Audio must be mono, 16-bit PCM.")

        while True:
            frame = audio.readframes(480)

            if len(frame) < frame_bytes:
                break

            is_speech = vad.is_speech(frame, 16000)

            total_frames += 1
            if is_speech:
                speech_frames += 1

    speech_seconds = speech_frames * frame_ms / 1000
    total_seconds = total_frames * frame_ms / 1000

    return speech_seconds, total_seconds, speech_frames, total_frames


def main():
    print("=== WebRTC VAD Test ===")

    files = find_recordings()
    recording = select_recording(files)

    if recording is None:
        return

    print(f"\nSelected: {recording.name}")

    wav_file = convert_to_wav(recording)

    try:
        speech, total, speech_frames, total_frames = analyze_vad(wav_file)

        percentage = (speech / total * 100) if total else 0

        print("\n=== VAD RESULTS ===")
        print(f"Total analyzed time: {total:.2f} seconds")
        print(f"Speech-active time: {speech:.2f} seconds")
        print(f"Non-speech time: {total - speech:.2f} seconds")
        print(f"Speech-active frames: {speech_frames}/{total_frames}")
        print(f"Speech activity: {percentage:.1f}%")

    finally:
        if wav_file.exists():
            wav_file.unlink()


if __name__ == "__main__":
    main()
