import os
import json
import requests
from openrouter_client import post_with_fallback
from pathlib import Path


# =========================
# CONFIGURATION
# =========================

TRANSCRIPTS_DIR = Path("transcripts")

API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
FALLBACK_MODEL = "nvidia/nemotron-3.5-lightning:free"


# =========================
# TRANSCRIPT SELECTION
# =========================

def find_transcripts():
    """Find saved transcript files."""

    if not TRANSCRIPTS_DIR.exists():
        return []

    return sorted(
        [
            file
            for file in TRANSCRIPTS_DIR.iterdir()
            if file.is_file()
            and file.suffix.lower() == ".txt"
        ],
        key=lambda file: file.name.lower()
    )


def select_transcript(files):
    """Allow the user to select a transcript."""

    if not files:
        print("No transcripts found.")
        return None

    print("\nAvailable transcripts:")

    for i, file in enumerate(files, start=1):
        print(f"{i}. {file.name}")

    while True:
        choice = input(
            "\nChoose a transcript number (or q to quit): "
        ).strip()

        if choice.lower() == "q":
            return None

        if choice.isdigit():
            index = int(choice)

            if 1 <= index <= len(files):
                return files[index - 1]

        print("Invalid choice.")


# =========================
# API
# =========================

def check_grammar(text):
    """Send transcript to the OpenRouter grammar checker."""

    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set."
        )

    system_prompt = """
You are a precise grammar checker for a speech-training application.

Analyze the user's transcript for actual grammatical errors.

Do NOT:
- rewrite sentences merely because they sound less natural
- criticize speaking style
- change slang unless it is grammatically incorrect
- invent errors
- penalize incomplete spoken phrases that are normal in conversation

For every genuine grammatical error, provide:
- the original text
- the corrected text
- the error type
- a short explanation

Also provide a corrected version of the entire transcript.

Return ONLY valid JSON using this structure:

{
  "errors": [
    {
      "original": "...",
      "correction": "...",
      "type": "...",
      "explanation": "..."
    }
  ],
  "corrected_text": "..."
}
"""

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": text,
            },
        ],
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    response = post_with_fallback(
        API_URL,
        fallback_model=FALLBACK_MODEL,
        headers=headers,
        json=payload,
        timeout=60,
    )

    if not response.ok:
        raise RuntimeError(
            f"OpenRouter API error "
            f"{response.status_code}: "
            f"{response.text}"
        )

    data = response.json()

    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        raise RuntimeError(
            "Unexpected response from OpenRouter."
        )

    # Remove possible Markdown code fences.
    content = content.strip()

    if content.startswith("```"):
        content = content.replace("```json", "", 1)
        content = content.replace("```", "", 1)
        content = content.strip()

    try:
        return json.loads(content)

    except json.JSONDecodeError:
        print("\nRaw model response:")
        print(content)

        raise RuntimeError(
            "The model did not return valid JSON."
        )


# =========================
# DISPLAY
# =========================

def display_results(result):
    """Display grammar analysis."""

    errors = result.get("errors", [])
    corrected_text = result.get(
        "corrected_text",
        ""
    )

    print("\n" + "=" * 50)
    print("             GRAMMAR ANALYSIS")
    print("=" * 50)

    print(
        f"\nGrammar errors found: {len(errors)}"
    )

    if errors:

        print("\n--- Errors ---")

        for i, error in enumerate(
            errors,
            start=1
        ):

            print(f"\n{i}. {error.get('type', 'Unknown')}")

            print(
                f"   Original: "
                f"{error.get('original', '')}"
            )

            print(
                f"   Correction: "
                f"{error.get('correction', '')}"
            )

            print(
                f"   Explanation: "
                f"{error.get('explanation', '')}"
            )

    else:

        print(
            "\nNo grammatical errors detected."
        )

    print("\n--- Corrected Transcript ---")
    print(corrected_text)

    print("\n" + "=" * 50)


# =========================
# MAIN
# =========================

def main():

    print(
        "=== Voice Trainer: Grammar Checker ==="
    )

    try:

        transcripts = find_transcripts()

        transcript_file = select_transcript(
            transcripts
        )

        if transcript_file is None:
            return

        text = transcript_file.read_text(
            encoding="utf-8"
        ).strip()

        if not text:
            print("Transcript is empty.")
            return

        print(
            f"\nSelected: {transcript_file.name}"
        )

        print("\nChecking grammar...")

        result = check_grammar(text)

        display_results(result)

    except Exception as error:

        print(
            f"\nGrammar checking error: {error}"
        )


if __name__ == "__main__":
    main()
