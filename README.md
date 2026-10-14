# Eloquify — Speak. Analyze. Improve.

Eloquify is a local-first Termux/Python speech-training application for Senior High School students. This package adds the first integrated version of **Personal Practice** while retaining the existing recording, transcription, analysis, profile, and browsing modules.

## What changed

- `main.py`: recording now accepts a filename prefix so practice answers can be saved as `practice_<attempt-id>_<timestamp>.m4a`; ordinary recordings retain the `recording_...` prefix.
- `speech_analyzer.py`: saves an atomic structured snapshot to `data/latest_analysis.json` after analysis. CEFR-inspired output is explicitly marked provisional and nonofficial.
- `practice_engine.py`: generates one or two evidence-based exercises from the latest analysis, records and transcribes an answer, stores history, and supports retrying failed transcription/evaluation while preserving prior successful evaluations.
- `practice_evaluator.py`: validates the evaluation structure and ensures returned success criteria match the original exercise.
- `ui.py`: adds **PERSONAL PRACTICE** to the dashboard.
- `tests/test_practice_validation.py`: offline tests for exercise/evaluation validation and JSON-safe serialization.

## Requirements

- Python 3.13 (or a compatible Python 3 version)
- Termux and the Termux:API Android app
- Python packages already used by the application: `rich`, `requests`, and `webrtcvad`
- FFmpeg (`pkg install ffmpeg` in Termux)
- Termux:API package (`pkg install termux-api`) and Android microphone permission
- `DEEPGRAM_API_KEY` for transcription/filler detection
- `OPENROUTER_API_KEY` for grammar feedback, recommendations, and practice generation/evaluation

Set API keys as environment variables in Termux, never in source code:

```bash
export DEEPGRAM_API_KEY="your_key_here"
export OPENROUTER_API_KEY="your_key_here"
```

To keep them across Termux sessions, use a private shell configuration file. Do not share it or include it in ZIP packages.

## Install / update

1. Back up the current `~/voice_trainer` folder.
2. Extract the replacement source files into `~/voice_trainer`, preserving your own `data/profile.json`, recordings, transcripts, and reports.
3. Install missing dependencies if needed:

   ```bash
   pkg install python ffmpeg termux-api
   python -m pip install rich requests webrtcvad
   ```

4. Run the app from Termux:

   ```bash
   cd ~/voice_trainer
   python ui.py
   ```

5. Record a speech, transcribe it, and run **ANALYZE SPEECH** before selecting **PERSONAL PRACTICE**. The latest analysis snapshot is stored at `data/latest_analysis.json`.

## Local storage

- `data/profile.json`: existing student profile
- `data/latest_analysis.json`: latest structured analysis used to generate exercises
- `data/practice_history.json`: generated exercises and practice attempts, including recording paths, transcripts, and evaluations
- `recordings/`: source and practice audio files
- `transcripts/`: saved transcripts and human-readable analysis reports

This package intentionally excludes personal profile data, saved transcripts, and recordings. Keep those files on your device and back them up separately if desired.

## Validation and limitations

Run offline checks from the project directory:

```bash
python -m py_compile main.py ui.py profile_manager.py transcript.py speech_analyzer.py practice_engine.py practice_evaluator.py
python -m unittest discover -s tests -v
```

These checks validate Python syntax and local validation logic only. They do **not** verify microphone behavior, Android permissions, FFmpeg behavior on the device, network connectivity, or real API responses. Test recording and stopping on the actual Android/Termux device.

Practice feedback is a learning aid. Pauses are not automatically errors, speaking-rate values have no universal ideal threshold, transcript text cannot assess pronunciation, and the provisional language estimate is not an official CEFR result. The app does not claim objective improvement unless repeated performance is measured under comparable conditions.
