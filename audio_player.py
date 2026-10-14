
import subprocess
from pathlib import Path

RECORDINGS_DIR = Path("recordings")

SUPPORTED_FORMATS = {
    ".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac"
}


def list_audio_files():
    return sorted(
        file for file in RECORDINGS_DIR.iterdir()
        if file.is_file() and file.suffix.lower() in SUPPORTED_FORMATS
    )


def play_audio(path):
    print(f"\nPlaying: {path.name}")

    try:
        subprocess.run(
            ["termux-media-player", "play", str(path)],
            check=True
        )
    except FileNotFoundError:
        print("Termux:API is not installed or unavailable.")
    except subprocess.CalledProcessError:
        print("Could not play this audio file.")


def stop_audio():
    subprocess.run(
        ["termux-media-player", "stop"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    print("Playback stopped.")


def main():
    if not RECORDINGS_DIR.exists():
        print("The recordings folder does not exist.")
        return

    while True:
        files = list_audio_files()

        if not files:
            print("No audio files found.")
            return

        print("\n=== Audio Tester ===")
        for index, file in enumerate(files, start=1):
            print(f"{index}. {file.name}")

        print("\nP. Play a file")
        print("S. Stop playback")
        print("R. Refresh file list")
        print("Q. Quit")

        choice = input("\nChoose an option: ").strip().lower()

        if choice == "q":
            stop_audio()
            print("Exiting.")
            break

        elif choice == "s":
            stop_audio()

        elif choice == "r":
            continue

        elif choice == "p":
            selection = input("Enter the file number: ").strip()

            if selection.isdigit():
                number = int(selection)

                if 1 <= number <= len(files):
                    play_audio(files[number - 1])
                else:
                    print("Invalid file number.")
            else:
                print("Enter a valid number.")

        else:
            print("Invalid option.")


if __name__ == "__main__":
    main()
