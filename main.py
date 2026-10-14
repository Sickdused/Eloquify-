#!/usr/bin/env python3
"""Eloquify recording screen for Termux.

Start recording with the Record button and stop only when the user explicitly
chooses Stop. Requires Termux:API, the Termux:API Android app, and Rich.
"""
from __future__ import annotations

import json
import os
import re
import select
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from rich.align import Align
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt
    from rich.text import Text
    from rich.live import Live
    from rich import box
except ImportError:
    print("Eloquify needs Rich. Install it with: python -m pip install rich")
    raise SystemExit(1)


PROJECT_DIR = Path.home() / "voice_trainer"
RECORDINGS_DIR = PROJECT_DIR / "recordings"
RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
console = Console()


def format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def recording_card(status: str = "READY", elapsed: float = 0, filename: str = "") -> Panel:
    """Build the centered microphone card."""
    if status == "RECORDING":
        mic = Text("🎙", style="bold red", justify="center")
        title = Text("RECORDING IN PROGRESS", style="bold red", justify="center")
        timer = Text(format_duration(elapsed), style="bold white", justify="center")
        hint = Text("Speak naturally. Your recording will continue until you stop it.", style="white", justify="center")
        action = Text("  [ S ]  STOP RECORDING  ", style="bold white on red", justify="center")
        body = Text()
        body.append_text(mic)
        body.append("\n\n")
        body.append_text(title)
        body.append("\n\n")
        body.append_text(timer)
        body.append("\n\n")
        body.append_text(hint)
        body.append("\n\n")
        body.append_text(action)
    else:
        mic = Text("🎙", style="bold bright_magenta", justify="center")
        title = Text("READY TO SPEAK?", style="bold bright_white", justify="center")
        subtitle = Text("Practice your speech at your own pace.", style="white", justify="center")
        action = Text("  [ R ]  RECORD  ", style="bold white on bright_magenta", justify="center")
        foot = Text("No time limit • Stop only when you are finished", style="dim", justify="center")
        body = Text()
        body.append_text(mic)
        body.append("\n\n")
        body.append_text(title)
        body.append("\n\n")
        body.append_text(subtitle)
        body.append("\n\n")
        body.append_text(action)
        body.append("\n\n")
        body.append_text(foot)
    return Panel(
        Align.center(body, vertical="middle"),
        title="[bold bright_cyan]ELOQUIFY  •  SPEAK. ANALYZE. IMPROVE.[/bold bright_cyan]",
        subtitle=f"[dim]{filename}[/dim]" if filename else "[dim]Speech Recording Studio[/dim]",
        border_style="bright_magenta" if status != "RECORDING" else "red",
        box=box.ROUNDED,
        padding=(1, 3),
        expand=True,
    )


def header() -> None:
    console.print()
    console.print(Panel(
        Align.center(Text("E L O Q U I F Y", style="bold bright_magenta")),
        subtitle="Your personal speech practice space",
        border_style="bright_cyan",
        box=box.DOUBLE,
    ))


def termux_api_ready() -> bool:
    if shutil.which("termux-microphone-record") is None:
        console.print("[bold red]Termux:API command not found.[/bold red]")
        console.print("Install the package with: [cyan]pkg install termux-api[/cyan]")
        console.print("Also install the Termux:API Android app and grant microphone permission.")
        return False
    return True


def start_recording(output_file: Path) -> bool:
    command = [
        "termux-microphone-record",
        "-f", str(output_file),
        "-l", "0",                 # zero means unlimited duration
        "-e", "aac",
        "-b", "64",                # kbps
        "-r", "16000",
        "-c", "1",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired) as exc:
        console.print(f"[red]Could not start recording: {exc}[/red]")
        return False

    output = (result.stdout + "\n" + result.stderr).strip()
    if result.returncode != 0 or "ERROR" in output.upper():
        console.print("[bold red]Recording did not start.[/bold red]")
        if output:
            console.print(output)
        console.print("Check microphone permission in Android settings.")
        return False
    return True


def stop_recording() -> None:
    try:
        result = subprocess.run(
            ["termux-microphone-record", "-q"],
            capture_output=True, text=True, timeout=15
        )
        output = (result.stdout + "\n" + result.stderr).strip()
        if result.returncode != 0 and "no recording to stop" not in output.lower():
            console.print(f"[yellow]Stop command returned a warning:[/yellow] {output}")
    except (OSError, subprocess.TimeoutExpired) as exc:
        console.print(f"[yellow]Could not confirm recording stop: {exc}[/yellow]")


def wait_for_stop(started_at: float, filename: str) -> None:
    """Keep the card updating while accepting S or Enter to explicitly stop."""
    console.print("[dim]Press [bold]S[/bold] then Enter, or just Enter, when you want to stop.[/dim]")
    try:
        with Live(recording_card("RECORDING", 0, filename), console=console,
                  refresh_per_second=4, screen=False) as live:
            while True:
                elapsed = time.monotonic() - started_at
                live.update(recording_card("RECORDING", elapsed, filename))
                ready, _, _ = select.select([sys.stdin], [], [], 0.25)
                if ready:
                    choice = sys.stdin.readline().strip().lower()
                    if choice in ("", "s", "stop", "q"):
                        break
                    console.print("[dim]Still recording — press Enter or type S then Enter to stop.[/dim]")
    except (KeyboardInterrupt, EOFError):
        console.print("\n[yellow]Stop requested.[/yellow]")
    finally:
        stop_recording()


def record_session(prefix: str = "recording") -> Path | None:
    if not termux_api_ready():
        Prompt.ask("Press Enter to return", default="")
        return None

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_prefix = re.sub(r"[^A-Za-z0-9_-]+", "_", str(prefix)).strip("_") or "recording"
    output_file = RECORDINGS_DIR / f"{safe_prefix}_{stamp}.m4a"

    console.clear()
    header()
    console.print(recording_card())
    console.print("\n[bold bright_cyan]Ready when you are.[/bold bright_cyan]")
    Prompt.ask("Press Enter to start recording", default="")

    if not start_recording(output_file):
        Prompt.ask("Press Enter to return", default="")
        return None

    started_at = time.monotonic()
    wait_for_stop(started_at, output_file.name)

    # Give the recorder service a moment to finalize the container.
    time.sleep(0.8)
    if output_file.exists() and output_file.stat().st_size > 0:
        size_kb = output_file.stat().st_size / 1024
        console.print(Panel(
            f"[bold green]Recording saved![/bold green]\n\n"
            f"[white]{output_file.name}[/white]\n"
            f"Duration: [cyan]{format_duration(time.monotonic() - started_at)}[/cyan]\n"
            f"File size: [cyan]{size_kb:.1f} KB[/cyan]\n\n"
            f"[dim]Saved in: {RECORDINGS_DIR}[/dim]",
            title="[bold green]✓ SESSION COMPLETE[/bold green]",
            border_style="green",
            box=box.ROUNDED,
        ))
        return output_file

    console.print("[bold yellow]The recording file was not found or is empty.[/bold yellow]")
    console.print("Check Termux:API microphone permission and try again.")
    return None


def analyze_audio(path: Path) -> None:
    """Show basic file details; does not change the saved recording."""
    console.print(Panel(f"[bold]Audio file[/bold]\n{path}", border_style="cyan"))
    size = path.stat().st_size if path.exists() else 0
    console.print(f"Size: {size:,} bytes")
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        console.print("[dim]Install ffmpeg to view codec details: pkg install ffmpeg[/dim]")
        return
    try:
        result = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries",
             "format=duration,format_name:stream=codec_name,sample_rate,channels",
             "-of", "json", str(path)],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            info: dict[str, Any] = json.loads(result.stdout or "{}")
            console.print_json(json.dumps(info, indent=2))
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        console.print("[dim]Could not read codec details.[/dim]")


def main() -> None:
    while True:
        console.clear()
        header()
        console.print(recording_card())
        console.print("\n[bold bright_white]Choose an action[/bold bright_white]")
        console.print("  [bright_magenta]R[/bright_magenta]  Record speech")
        console.print("  [bright_cyan]L[/bright_cyan]  List saved recordings")
        console.print("  [bright_cyan]Q[/bright_cyan]  Return / exit")
        choice = Prompt.ask("\n[bold bright_magenta]Eloquify[/bold bright_magenta]", choices=["r", "l", "q"], default="r").lower()

        if choice == "r":
            saved = record_session()
            if saved:
                analyze_audio(saved)
            Prompt.ask("\nPress Enter to return to the recording studio", default="")
        elif choice == "l":
            files = sorted(RECORDINGS_DIR.glob("*.m4a"), key=lambda p: p.stat().st_mtime, reverse=True)
            if not files:
                console.print("[yellow]No recordings yet.[/yellow]")
            else:
                console.print(Panel("\n".join(f"{i}. {p.name}  ({p.stat().st_size / 1024:.1f} KB)" for i, p in enumerate(files[:30], 1)),
                                    title="Saved Recordings", border_style="cyan"))
            Prompt.ask("Press Enter to return", default="")
        else:
            console.print("[bright_magenta]Keep practicing — every attempt counts![/bright_magenta]")
            break


if __name__ == "__main__":
    main()
