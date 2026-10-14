import os
import subprocess
import wave
import math
import json
import uuid
import time
from datetime import datetime
from pathlib import Path
from contextlib import nullcontext

import requests

try:
    from rich.console import Console
    _console = Console()
except ImportError:
    _console = None


# =========================
# CONFIGURATION
# =========================

PROJECT_DIR = Path(__file__).resolve().parent
RECORDINGS_DIR = PROJECT_DIR / "recordings"
TRANSCRIPTS_DIR = PROJECT_DIR / "transcripts"
LAST_TRANSCRIPTION_FILE = PROJECT_DIR / "last_transcription.json"

SAMPLE_RATE = 16000
DEEPGRAM_API_URL = "https://api.deepgram.com/v1/listen"
DEEPGRAM_MODEL = "nova-3"
DEEPGRAM_TIMEOUT = (15, 180)  # connect timeout, response timeout
DEEPGRAM_MAX_ATTEMPTS = 3

SUPPORTED_FORMATS = {
    ".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac",
}


# =========================
# FILE SELECTION
# =========================

def list_audio_files():
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(
        (
            path for path in RECORDINGS_DIR.iterdir()
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_FORMATS
            and not path.name.startswith("_temp")
        ),
        key=lambda path: (path.stat().st_mtime, path.name.lower()),
        reverse=True,
    )


def select_audio_file():
    files = list_audio_files()
    if not files:
        print("\nNo supported audio files found.")
        print(f"Place recordings in: {RECORDINGS_DIR.resolve()}")
        return None

    print("\nAVAILABLE RECORDINGS (newest first)")
    print("-" * 64)
    for index, path in enumerate(files, start=1):
        try:
            size_mb = path.stat().st_size / (1024 * 1024)
            modified = datetime.fromtimestamp(path.stat().st_mtime).strftime(
                "%Y-%m-%d %H:%M"
            )
            print(f"{index:>2}. {path.name}")
            print(f"    {modified}  |  {size_mb:.2f} MB")
        except OSError:
            print(f"{index:>2}. {path.name}")

    while True:
        choice = input("\nEnter the number of the EXACT recording (or q to quit): ").strip()
        if choice.lower() == "q":
            return None
        if choice.isdigit() and 1 <= int(choice) <= len(files):
            selected = files[int(choice) - 1].resolve()
            print("\nYou selected:")
            print(f"  {selected.name}")
            print(f"  {selected}")
            confirm = input("Transcribe this exact file? (Y/n): ").strip().lower()
            if confirm in ("", "y", "yes"):
                if not selected.is_file() or selected.stat().st_size == 0:
                    print("That file is missing or empty. Please choose again.")
                    continue
                return selected
            print("\nSelection cancelled. Choose a recording again.")
            continue
        print("Invalid selection. Enter one of the listed numbers or q.")


# =========================
# AUDIO CONVERSION / VALIDATION
# =========================

def convert_audio(source, destination):
    """Convert the exact selected source into mono 16 kHz PCM WAV."""
    source = Path(source).resolve()
    destination = Path(destination).resolve()

    if not source.is_file():
        print(f"\nSelected recording does not exist: {source}")
        return False

    # Never allow an old output file to survive a failed conversion.
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.unlink(missing_ok=True)
    except OSError as error:
        print(f"\nCould not prepare temporary audio file: {error}")
        return False

    command = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-y", "-i", str(source), "-map", "0:a:0", "-vn",
        "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le",
        str(destination),
    ]

    try:
        result = subprocess.run(
            command, capture_output=True, text=True, check=False, timeout=180
        )
    except FileNotFoundError:
        print("FFmpeg was not found. Install it with: pkg install ffmpeg")
        return False
    except subprocess.TimeoutExpired:
        print("Audio conversion timed out.")
        destination.unlink(missing_ok=True)
        return False

    if result.returncode != 0:
        print("\nFFmpeg conversion failed.")
        if result.stderr:
            print(result.stderr.strip())
        destination.unlink(missing_ok=True)
        return False

    try:
        if not destination.is_file() or destination.stat().st_size <= 44:
            print("Conversion produced an empty or invalid WAV file.")
            destination.unlink(missing_ok=True)
            return False
        with wave.open(str(destination), "rb") as wav:
            valid = (
                wav.getframerate() == SAMPLE_RATE
                and wav.getnchannels() == 1
                and wav.getsampwidth() == 2
                and wav.getnframes() > 0
            )
        if not valid:
            print("Converted WAV failed format validation.")
            destination.unlink(missing_ok=True)
            return False
    except (OSError, wave.Error) as error:
        print(f"Could not validate converted audio: {error}")
        destination.unlink(missing_ok=True)
        return False

    return True


def inspect_audio(path):
    try:
        with wave.open(str(path), "rb") as audio:
            sample_rate = audio.getframerate()
            channels = audio.getnchannels()
            sample_width = audio.getsampwidth()
            frame_count = audio.getnframes()
            raw_audio = audio.readframes(frame_count)

        duration = frame_count / sample_rate if sample_rate else 0
        print("\nCONVERTED AUDIO CHECK")
        print("-" * 40)
        print(f"Duration:    {duration:.2f} seconds")
        print(f"Sample rate: {sample_rate} Hz")
        print(f"Channels:    {channels}")
        print(f"Sample size: {sample_width * 8} bits")

        if sample_width != 2:
            print("Level analysis skipped: expected 16-bit PCM.")
            return
        import array
        samples = array.array("h")
        samples.frombytes(raw_audio)
        if os.sys.byteorder != "little":
            samples.byteswap()
        if not samples:
            print("No audio samples found.")
            return
        peak = max(abs(sample) for sample in samples)
        rms = math.sqrt(sum(sample * sample for sample in samples) / len(samples))
        peak_dbfs = 20 * math.log10(peak / 32768.0) if peak else float("-inf")
        rms_dbfs = 20 * math.log10(rms / 32768.0) if rms else float("-inf")
        print(f"Peak level:  {peak_dbfs:.2f} dBFS")
        print(f"RMS level:   {rms_dbfs:.2f} dBFS")
    except (wave.Error, OSError, ValueError) as error:
        print(f"Audio inspection failed: {error}")


# =========================
# DEEPGRAM
# =========================

def transcribe_audio(audio_path):
    """Transcribe a validated WAV, automatically retry transient failures, and allow manual retry."""
    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        print('\nDEEPGRAM_API_KEY is not set. Set it with:')
        print('export DEEPGRAM_API_KEY="your_api_key_here"')
        return None

    audio_path = Path(audio_path).resolve()
    if not audio_path.is_file() or audio_path.stat().st_size <= 44:
        print(f"\nRefusing to transcribe missing or empty audio: {audio_path}")
        return None

    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "audio/wav",
    }
    params = {
        "model": DEEPGRAM_MODEL,
        "filler_words": "true",
        "punctuate": "true",
        "utterances": "false",
        "smart_format": "true",
    }

    # The outer loop is the user's manual retry. Each round gets automatic retries first.
    while True:
        last_error = "Deepgram request did not complete."
        retryable = True
        for attempt in range(1, DEEPGRAM_MAX_ATTEMPTS + 1):
            status_context = (
                _console.status(
                    f"[cyan]Uploading audio to Deepgram · attempt {attempt}/{DEEPGRAM_MAX_ATTEMPTS}…[/cyan]",
                    spinner="dots",
                ) if _console else nullcontext()
            )
            try:
                print(f"\nSending validated audio to Deepgram: {audio_path.name}")
                # Read bytes afresh on every attempt so retries resend the complete WAV.
                audio_bytes = audio_path.read_bytes()
                with status_context:
                    response = requests.post(
                        DEEPGRAM_API_URL,
                        headers=headers,
                        params=params,
                        data=audio_bytes,
                        timeout=DEEPGRAM_TIMEOUT,
                    )

                if response.status_code != 200:
                    last_error = f"HTTP status: {response.status_code}\n"
                    try:
                        last_error += str(response.json())
                    except ValueError:
                        last_error += response.text[:2000]
                    retryable = response.status_code in (408, 425, 429) or response.status_code >= 500
                    print("\nDeepgram API request failed.")
                    print(last_error)
                    if retryable and attempt < DEEPGRAM_MAX_ATTEMPTS:
                        wait_seconds = min(2 ** (attempt - 1), 4)
                        print(f"Temporary Deepgram error. Retrying in {wait_seconds} seconds…")
                        time.sleep(wait_seconds)
                        continue
                    break

                result = response.json()
                channels = result.get("results", {}).get("channels", [])
                if not channels:
                    last_error = "Deepgram returned no audio channels."
                    retryable = False
                    break
                alternatives = channels[0].get("alternatives", [])
                if not alternatives:
                    last_error = "Deepgram returned no transcript alternatives."
                    retryable = False
                    break
                transcript = alternatives[0].get("transcript", "").strip()
                if not transcript:
                    last_error = "Deepgram returned an empty transcript."
                    retryable = False
                    break
                return transcript

            except requests.Timeout as error:
                last_error = f"Deepgram request timed out: {error}"
                retryable = True
            except requests.RequestException as error:
                last_error = f"Deepgram network/upload error: {error}"
                retryable = True
            except OSError as error:
                last_error = f"Could not read the audio file: {error}"
                retryable = False
                break
            except (KeyError, IndexError, ValueError, TypeError) as error:
                last_error = f"Could not parse the Deepgram API response: {error}"
                retryable = False
                break

            print(f"\n{last_error}")
            if retryable and attempt < DEEPGRAM_MAX_ATTEMPTS:
                wait_seconds = min(2 ** (attempt - 1), 4)
                print(f"Temporary Deepgram error. Retrying in {wait_seconds} seconds…")
                time.sleep(wait_seconds)
                continue
            break

        print("\nTranscription could not be completed. Your recording is preserved.")
        choice = input("Enter r to retry Deepgram, or press Enter to return: ").strip().lower()
        if choice == "r":
            print("\nRetrying transcription…")
            continue
        return None


# =========================
# DISPLAY / SAVE
# =========================

def display_transcript(text):
    print("\n" + "=" * 60)
    print("TRANSCRIPTION — DEEPGRAM")
    print("=" * 60)
    print(text)
    print("=" * 60)


def save_transcript(source, text):
    """Save a uniquely named transcript and an exact source-file manifest."""
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    source = Path(source).resolve()

    # Unique filename avoids collisions when recordings share the same stem,
    # and preserves earlier transcripts instead of silently overwriting them.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:6]
    output_path = TRANSCRIPTS_DIR / f"{source.stem}_{timestamp}_{unique_id}.txt"
    manifest = {
        "recording": str(source),
        "transcript": str(output_path.resolve()),
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": DEEPGRAM_MODEL,
    }

    try:
        if not source.is_file():
            print(f"\nOriginal recording no longer exists: {source}")
            return None
        output_path.write_text(text.strip() + "\n", encoding="utf-8")
        LAST_TRANSCRIPTION_FILE.write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        print(f"\nTranscript saved: {output_path.name}")
        print(f"Source recording: {source.name}")
        print(f"Exact source path: {source}")
        print(f"Pairing manifest: {LAST_TRANSCRIPTION_FILE.name}")
        return output_path
    except OSError as error:
        print(f"\nCould not save transcript or recording link: {error}")
        try:
            output_path.unlink(missing_ok=True)
        except OSError:
            pass
        return None


# =========================
# MAIN
# =========================

def main():
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

    source = select_audio_file()
    if source is None:
        print("\nExiting.")
        return

    # Unique temporary path prevents stale files and collisions across runs.
    temp_audio = RECORDINGS_DIR / f"_temp_transcription_{uuid.uuid4().hex}.wav"
    try:
        print(f"\nSelected source (absolute path):\n{source}")
        print("\nConverting this recording to a temporary WAV...")

        if not convert_audio(source, temp_audio):
            print("\nTranscription stopped because conversion failed.")
            return

        inspect_audio(temp_audio)
        transcript = transcribe_audio(temp_audio)
        if transcript is None:
            print("\nTranscription was unsuccessful. No transcript was saved.")
            return

        display_transcript(transcript)
        print(f"\nSource audio: {source.name}")
        save_choice = input("Save this transcript for this recording? (Y/n): ").strip().lower()
        if save_choice in ("", "y", "yes"):
            save_transcript(source, transcript)
        else:
            print("\nTranscript was not saved.")
    finally:
        try:
            temp_audio.unlink(missing_ok=True)
        except OSError as error:
            print(f"Could not remove temporary audio: {error}")


if __name__ == "__main__":
    main()
