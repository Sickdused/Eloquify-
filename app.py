#!/usr/bin/env python3
"""Eloquify Rich terminal dashboard — replaces the Flask overview."""

from pathlib import Path
import subprocess
import sys
import shutil

from rich import box
from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

PROJECT_DIR = Path.home() / "voice_trainer"
RECORDINGS_DIR = PROJECT_DIR / "recordings"
RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

AUDIO_EXTENSIONS = {".m4a", ".wav", ".mp3", ".aac", ".ogg"}
console = Console()


def get_recordings():
    return sorted(
        [
            f for f in RECORDINGS_DIR.iterdir()
            if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS
        ],
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )


def header():
    title = Text("E L O Q U I F Y", style="bold bright_magenta")
    subtitle = Text("S P E A K  •  A N A L Y Z E  •  I M P R O V E",
                    style="bright_cyan")

    content = Text()
    content.append_text(title)
    content.append("\n")
    content.append_text(subtitle)

    console.print(Panel(
        Align.center(content),
        border_style="bright_cyan",
        box=box.DOUBLE,
        padding=(1, 2),
    ))


def dashboard():
    console.clear()
    header()

    recordings = get_recordings()

    console.print(Panel(
        "[bold white]Your speaking workspace[/bold white]\n"
        "[dim]Practice. Review. Improve your communication.[/dim]",
        title="[bold bright_magenta]DASHBOARD[/bold bright_magenta]",
        border_style="bright_magenta",
        box=box.ROUNDED,
        padding=(1, 2),
    ))

    stats = Table.grid(expand=True, padding=(1, 2))
    stats.add_column(ratio=1)
    stats.add_column(ratio=1)

    stats.add_row(
        Panel(
            f"[bold bright_white]{len(recordings)}[/bold bright_white]\n"
            "[bright_cyan]RECORDINGS[/bright_cyan]",
            border_style="bright_cyan",
            box=box.ROUNDED,
        ),
        Panel(
            "[bold bright_white]—[/bold bright_white]\n"
            "[bright_magenta]ANALYSIS REPORTS[/bright_magenta]\n"
            "[dim]Not connected yet[/dim]",
            border_style="bright_magenta",
            box=box.ROUNDED,
        ),
    )
    console.print(stats)

    recent = Table(
        title="[bold bright_cyan]RECENT RECORDINGS[/bold bright_cyan]",
        box=box.ROUNDED,
        border_style="cyan",
        expand=True,
    )
    recent.add_column("#", style="bright_magenta", width=4)
    recent.add_column("Filename", style="white")
    recent.add_column("Size", style="bright_cyan", justify="right")

    for i, file in enumerate(recordings[:5], 1):
        size = file.stat().st_size / 1024
        recent.add_row(str(i), file.name, f"{size:.1f} KB")

    if not recordings:
        recent.add_row("-", "No recordings yet", "-")

    console.print(recent)


def record_speech():
    recorder = PROJECT_DIR / "main.py"

    if not recorder.exists():
        console.print(f"[red]Recording UI not found: {recorder}[/red]")
        Prompt.ask("Press Enter to return", default="")
        return

    console.print("[bright_cyan]Opening recording interface...[/bright_cyan]")

    try:
        subprocess.run(
            [sys.executable, str(recorder)],
            cwd=str(PROJECT_DIR),
            check=False,
        )
    except OSError as exc:
        console.print(f"[bold red]Could not launch recorder:[/bold red] {exc}")
        Prompt.ask("Press Enter to return", default="")


def analysis():
    console.clear()
    header()

    recordings = get_recordings()

    table = Table(
        title="[bold bright_cyan]SELECT A RECORDING[/bold bright_cyan]",
        box=box.ROUNDED,
        border_style="cyan",
        expand=True,
    )
    table.add_column("#", style="bright_magenta", width=4)
    table.add_column("Filename")
    table.add_column("Size", justify="right")

    for i, file in enumerate(recordings, 1):
        table.add_row(
            str(i),
            file.name,
            f"{file.stat().st_size / 1024:.1f} KB",
        )

    if not recordings:
        table.add_row("-", "No recordings available", "-")
        console.print(table)
        Prompt.ask("Press Enter to return", default="")
        return

    console.print(table)

    choice = Prompt.ask(
        "Recording number, or B to go back",
        default="b",
    ).lower()

    if choice == "b":
        return

    if not choice.isdigit() or not 1 <= int(choice) <= len(recordings):
        console.print("[red]Invalid selection.[/red]")
        Prompt.ask("Press Enter to return", default="")
        return

    selected = recordings[int(choice) - 1]

    console.print(Panel(
        f"[bold]File:[/bold] {selected.name}\n"
        f"[bold]Location:[/bold] {selected}\n"
        f"[bold]Size:[/bold] {selected.stat().st_size:,} bytes",
        title="[bold green]RECORDING DETAILS[/bold green]",
        border_style="green",
        box=box.ROUNDED,
    ))

    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        result = subprocess.run(
            [
                ffprobe, "-v", "error",
                "-show_entries",
                "format=duration,format_name:stream=codec_name,sample_rate,channels",
                "-of", "default=noprint_wrappers=1",
                str(selected),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            console.print(Panel(
                result.stdout.strip(),
                title="Audio metadata",
                border_style="cyan",
            ))
    else:
        console.print("[dim]Install ffmpeg for audio metadata: pkg install ffmpeg[/dim]")

    console.print(
        "\n[yellow]This screen displays audio metadata only. "
        "Your AI transcription and speech analysis are not connected here yet.[/yellow]"
    )
    Prompt.ask("Press Enter to return", default="")


def progress():
    console.clear()
    header()

    recordings = get_recordings()

    console.print(Panel(
        f"[bold]Saved recordings:[/bold] {len(recordings)}\n\n"
        "[dim]Detailed speech scores and progress history can be added "
        "when connected to your existing analysis data.[/dim]",
        title="[bold bright_magenta]PROGRESS & HISTORY[/bold bright_magenta]",
        border_style="bright_magenta",
        box=box.ROUNDED,
    ))

    Prompt.ask("Press Enter to return", default="")


def main():
    while True:
        dashboard()

        menu = Table(
            title="[bold white]MAIN MENU[/bold white]",
            box=box.ROUNDED,
            border_style="bright_magenta",
            show_header=False,
            expand=True,
        )
        menu.add_column("Key", style="bold bright_magenta", width=8)
        menu.add_column("Action", style="white")

        menu.add_row("1", "Record Speech")
        menu.add_row("2", "Analyze Recording")
        menu.add_row("3", "Progress & History")
        menu.add_row("4", "Refresh Dashboard")
        menu.add_row("Q", "Quit")

        console.print(menu)

        choice = Prompt.ask(
            "[bold bright_cyan]Eloquify[/bold bright_cyan]",
            choices=["1", "2", "3", "4", "q"],
            default="4",
        ).lower()

        if choice == "1":
            record_speech()
        elif choice == "2":
            analysis()
        elif choice == "3":
            progress()
        elif choice == "4":
            continue
        else:
            console.print(
                "\n[bold bright_magenta]"
                "Keep practicing — every attempt counts!"
                "[/bold bright_magenta]"
            )
            break


if __name__ == "__main__":
    main()
