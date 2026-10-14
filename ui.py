#!/usr/bin/env python3
"""Eloquify: local-first, mobile-friendly Termux interface."""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from profile_manager import (
    PROJECT_DIR,
    LANGUAGES,
    PROFICIENCY_LEVELS,
    load_profile,
    save_profile,
)

RECORDINGS_DIR = PROJECT_DIR / "recordings"
TRANSCRIPTS_DIR = PROJECT_DIR / "transcripts"

SUPPORTED_AUDIO = {
    ".wav", ".mp3", ".m4a", ".mp4",
    ".flac", ".ogg", ".webm", ".aac",
}

theme = Theme({
    "cyan": "bold bright_cyan",
    "violet": "bold bright_magenta",
    "green": "bold bright_green",
    "yellow": "bold yellow",
    "muted": "dim white",
})
console = Console(theme=theme, soft_wrap=True)


def clear():
    console.clear()


def pause():
    console.print()
    Prompt.ask("[muted]Press Enter to continue[/muted]", default="")


def file_count(folder, pattern="*"):
    if not folder.exists():
        return 0
    return sum(1 for p in folder.glob(pattern) if p.is_file())


def format_size(size):
    if size < 1024:
        return f"{size} B"
    if size < 1024 ** 2:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 ** 2):.1f} MB"


def section_title(title, subtitle=""):
    console.print()
    console.print(f"[cyan]{title.upper()}[/cyan]")
    if subtitle:
        console.print(f"[muted]{subtitle}[/muted]")
    console.print()


def show_header():
    """Display the pixelated Eloquify logo centered in the terminal."""
    pixels = {
        "E": ("11111", "10000", "11110", "10000", "11111"),
        "L": ("10000", "10000", "10000", "10000", "11111"),
        "O": ("01110", "10001", "10001", "10001", "01110"),
        "Q": ("01110", "10001", "10001", "01110", "00011"),
        "U": ("10001", "10001", "10001", "10001", "01110"),
        "I": ("11111", "00100", "00100", "00100", "11111"),
        "F": ("11111", "10000", "11110", "10000", "10000"),
        "Y": ("10001", "10001", "01010", "00100", "00100"),
    }

    letters = "ELOQUIFY"
    logo_rows = []

    for row_index in range(5):
        line = Text()
        for index, letter in enumerate(letters):
            color = "bright_cyan" if index < 3 or index == 7 else "bright_magenta"
            for pixel in pixels[letter][row_index]:
                line.append("█" if pixel == "1" else " ", style=color)
            if index < len(letters) - 1:
                line.append(" ")
        logo_rows.append(Align.center(line, vertical="middle"))

    tagline = Align.center(Text(
        "S P E A K   •   A N A L Y Z E   •   I M P R O V E",
        style="dim white",
    ))

    console.print(Panel(
        Group(*logo_rows, Text(""), tagline),
        title="[bold bright_magenta] E L O Q U I F Y [/bold bright_magenta]",
        border_style="bright_magenta",
        box=box.DOUBLE,
        padding=(1, 2),
        expand=True,
    ))
def choose_from_list(title, options):
    """Prompt with numbered choices and return the selected option."""
    console.print(f"\n[bold bright_cyan]{title}[/bold bright_cyan]")
    for index, option in enumerate(options, 1):
        console.print(f"  [cyan]{index}.[/cyan] {option}")
    while True:
        choice = Prompt.ask(
            "[cyan]Choose a number[/cyan]",
            choices=[str(i) for i in range(1, len(options) + 1)],
        )
        return options[int(choice) - 1]


def profile_setup(existing=None):
    """Collect profile details. Existing profile values are used as edit defaults."""
    clear()
    show_header()
    console.print(Panel(
        Group(
            Text("LET'S GET TO KNOW YOU", style="bold white"),
            Text("Set up your profile to personalize your speaking practice.",
                 style="dim white"),
        ),
        title="[bold bright_magenta]YOUR ELOQUIFY PROFILE[/bold bright_magenta]",
        border_style="bright_cyan",
        box=box.ROUNDED,
        padding=(1, 2),
    ))

    default_name = existing.get("name", "") if existing else ""
    default_age = str(existing.get("age", 17)) if existing else "17"

    while True:
        name = Prompt.ask(
            "\n[bold bright_cyan]What should we call you?[/bold bright_cyan]",
            default=default_name or None,
        ).strip()
        if name:
            break
        console.print("[yellow]Please enter a name or nickname.[/yellow]")

    while True:
        age_text = Prompt.ask(
            "[bold bright_cyan]How old are you?[/bold bright_cyan]",
            default=default_age,
        ).strip()
        try:
            age = int(age_text)
            if 1 <= age <= 120:
                break
        except ValueError:
            pass
        console.print("[yellow]Enter a whole number from 1 to 120.[/yellow]")

    language = choose_from_list(
        "Which language are you most comfortable speaking?",
        LANGUAGES,
    )
    level = choose_from_list(
        "How would you describe your current speaking proficiency?",
        PROFICIENCY_LEVELS,
    )

    console.print()
    console.print(Panel(
        f"[bold white]Name:[/bold white] {name}\n"
        f"[bold white]Age:[/bold white] {age}\n"
        f"[bold white]Comfortable language:[/bold white] {language}\n"
        f"[bold white]Self-perceived speaking level:[/bold white] {level}",
        title="[bright_cyan]CONFIRM YOUR PROFILE[/bright_cyan]",
        border_style="bright_blue",
        box=box.ROUNDED,
    ))

    confirm = Prompt.ask(
        "[bold bright_cyan]Save this profile?[/bold bright_cyan]",
        choices=["y", "n"],
        default="y",
    )
    if confirm == "n":
        console.print("[yellow]Profile setup cancelled.[/yellow]")
        return None

    try:
        profile = save_profile(name, age, language, level)
    except (OSError, ValueError) as exc:
        console.print(f"[bold red]Could not save profile:[/bold red] {exc}")
        pause()
        return None

    return profile


def show_welcome(profile):
    clear()
    show_header()
    name = profile["name"]
    console.print(Panel(
        Group(
            Text(f"Welcome to Eloquify, {name}!", style="bold white"),
            Text("Your speaking journey starts here.", style="bright_cyan"),
            Text(
                "Practice speaking, understand your results, and work toward your goals.",
                style="dim white",
            ),
        ),
        title="[bold bright_magenta]WELCOME[/bold bright_magenta]",
        border_style="bright_magenta",
        box=box.ROUNDED,
        padding=(1, 2),
    ))

    details = Table.grid(expand=True, padding=(0, 1))
    details.add_column(style="bright_cyan", ratio=1)
    details.add_column(style="white", ratio=2)
    details.add_row("Comfortable language", profile["comfortable_language"])
    details.add_row("Self-rated speaking level", profile["perceived_speaking_level"])
    details.add_row("Age", str(profile["age"]))
    console.print(Panel(
        details,
        title="[bold white]YOUR PROFILE[/bold white]",
        border_style="bright_blue",
        box=box.ROUNDED,
    ))
    pause()


def show_status():
    recordings = file_count(RECORDINGS_DIR)
    transcripts = file_count(TRANSCRIPTS_DIR, "*.txt")
    status = Table.grid(expand=True, padding=(0, 1))
    status.add_column(justify="center")
    status.add_column(justify="center")
    status.add_row(
        f"[cyan]{recordings}[/cyan]\n[muted]Recordings[/muted]",
        f"[violet]{transcripts}[/violet]\n[muted]Transcripts[/muted]",
    )
    console.print(Panel(
        status,
        title="[bold white]YOUR WORKSPACE[/bold white]",
        border_style="bright_blue",
        box=box.ROUNDED,
    ))


def menu_card(number, title, description, color):
    return Panel(
        Group(
            Text(f"[ {number} ]", style=f"bold {color}"),
            Text(title, style="bold white"),
            Text(description, style="dim white"),
        ),
        border_style=color,
        box=box.ROUNDED,
        padding=(1, 2),
    )


def show_dashboard(profile):
    clear()
    show_header()
    console.print(
        f"[bold white]Hello, [bright_cyan]{profile['name']}[/bright_cyan]![/bold white] "
        "[muted]Ready to practice?[/muted]\n"
    )
    show_status()
    console.print("\\n[bold white]  SPEECH WORKSPACE[/bold white]\\n")

    cards = [
        ("1", "RECORD SPEECH", "Record your voice using the phone microphone", "bright_green"),
        ("2", "TRANSCRIBE SPEECH", "Convert a recording into text", "bright_cyan"),
        ("3", "ANALYZE SPEECH", "Metrics, fillers, grammar and suggestions", "bright_magenta"),
        ("4", "MY RECORDINGS", "Browse your audio files", "bright_blue"),
        ("5", "MY TRANSCRIPTS", "Read saved transcripts", "bright_yellow"),
        ("6", "ANALYSIS REPORTS", "Review your saved reports", "bright_green"),
        ("7", "MY PROFILE", "View or edit your local profile", "bright_cyan"),
        ("8", "PERSONAL PRACTICE", "Get a personalized speaking exercise and feedback", "bright_magenta"),
    ]

    grid = Table.grid(expand=True, padding=(0, 1))
    grid.add_column(ratio=1)
    grid.add_column(ratio=1)

    for index in range(0, len(cards), 2):
        grid.add_row(
            menu_card(*cards[index]),
            menu_card(*cards[index + 1]),
        )

    console.print(grid)
    console.print()
    console.print(Panel(
        "[bold white]0[/bold white]  Exit Eloquify",
        border_style="bright_black",
        box=box.ROUNDED,
        padding=(0, 2),
    ))
    console.print("\\n[muted]Enter an option number below.[/muted]")
def run_module(filename, title):
    script = PROJECT_DIR / filename
    clear()
    show_header()
    console.print(Panel(
        f"[bold white]{title}[/bold white]\n"
        "[muted]Running your existing speech module...[/muted]",
        border_style="bright_magenta",
        box=box.ROUNDED,
    ))
    console.print()
    if not script.is_file():
        console.print(f"[bold red]Missing file:[/bold red] {script}")
        pause()
        return
    try:
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(PROJECT_DIR),
            check=False,
        )
        console.print()
        if result.returncode == 0:
            console.print("[green]Process finished.[/green]")
        else:
            console.print(
                f"[yellow]Process exited with code {result.returncode}.[/yellow]"
            )
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation interrupted.[/yellow]")
    except OSError as exc:
        console.print(f"\n[bold red]Could not launch script:[/bold red] {exc}")
    pause()


def list_recordings():
    if not RECORDINGS_DIR.exists():
        return []
    return sorted(
        [p for p in RECORDINGS_DIR.iterdir()
         if p.is_file() and p.suffix.lower() in SUPPORTED_AUDIO],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def list_transcripts():
    if not TRANSCRIPTS_DIR.exists():
        return []
    return sorted(
        [p for p in TRANSCRIPTS_DIR.glob("*.txt") if p.is_file()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def list_reports():
    ignored_dirs = {
        ".git", ".venv", "venv", "__pycache__",
        "recordings", "transcripts", "data",
    }
    results = []
    for path in PROJECT_DIR.rglob("*_analysis.txt"):
        if not path.is_file():
            continue
        relative_parts = path.relative_to(PROJECT_DIR).parts
        if any(part in ignored_dirs for part in relative_parts):
            continue
        results.append(path)
    return sorted(results, key=lambda p: p.stat().st_mtime, reverse=True)


def browse_files(files, title, empty_message, open_as_text=False):
    clear()
    show_header()
    section_title(title, "Choose a file to open, or enter 0 to go back.")
    if not files:
        console.print(Panel(empty_message, border_style="yellow",
                            box=box.ROUNDED, padding=(1, 2)))
        pause()
        return
    table = Table(box=box.ROUNDED, border_style="bright_blue", expand=True)
    table.add_column("#", style="bright_cyan", justify="right", width=3)
    table.add_column("FILE", style="white", overflow="fold")
    table.add_column("SIZE", style="dim white", justify="right")
    for index, path in enumerate(files, 1):
        try:
            size = format_size(path.stat().st_size)
        except OSError:
            size = "?"
        table.add_row(str(index), path.name, size)
    console.print(table)
    console.print()
    while True:
        choice = Prompt.ask("[cyan]File number[/cyan]", default="0").strip()
        if choice == "0":
            return
        if choice.isdigit() and 1 <= int(choice) <= len(files):
            selected = files[int(choice) - 1]
            if open_as_text:
                display_text_file(selected)
            else:
                display_recording_info(selected)
            return
        console.print("[yellow]Enter a valid file number.[/yellow]")


def display_recording_info(path):
    clear()
    show_header()
    try:
        stat = path.stat()
        modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
        details = (
            f"[cyan]Filename[/cyan]\n{path.name}\n\n"
            f"[cyan]File size[/cyan]\n{format_size(stat.st_size)}\n\n"
            f"[cyan]Last modified[/cyan]\n{modified}\n\n"
            "[muted]Use Analyze Speech from the main menu to analyze recordings.[/muted]"
        )
        console.print(Panel(details, title="[bold white]RECORDING DETAILS[/bold white]",
                            border_style="bright_cyan", box=box.ROUNDED, padding=(1, 2)))
    except OSError as exc:
        console.print(f"[red]Could not read file details: {exc}[/red]")
    pause()


def display_text_file(path):
    clear()
    show_header()
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        console.print(f"[red]Could not open file: {exc}[/red]")
        pause()
        return
    console.print(Panel(
        content.strip() or "[muted]This file is empty.[/muted]",
        title=f"[bold white]{path.name}[/bold white]",
        border_style="bright_magenta", box=box.ROUNDED, padding=(1, 2),
    ))
    pause()


def manage_profile(profile):
    clear()
    show_header()
    console.print(Panel(
        f"[bold white]Name:[/bold white] {profile['name']}\n"
        f"[bold white]Age:[/bold white] {profile['age']}\n"
        f"[bold white]Comfortable language:[/bold white] {profile['comfortable_language']}\n"
        f"[bold white]Self-perceived speaking level:[/bold white] "
        f"{profile['perceived_speaking_level']}",
        title="[bold bright_cyan]MY PROFILE[/bold bright_cyan]",
        border_style="bright_magenta", box=box.ROUNDED, padding=(1, 2),
    ))
    choice = Prompt.ask(
        "[cyan]1[/cyan] Edit profile  [cyan]0[/cyan] Back",
        choices=["0", "1"], default="0",
    )
    if choice == "1":
        updated = profile_setup(existing=profile)
        if updated:
            profile = updated
            show_welcome(profile)
    return profile


def main():
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

    profile = load_profile()
    if profile is None:
        profile = profile_setup()
        if profile is None:
            console.print("[yellow]Eloquify needs a profile to continue. Exiting for now.[/yellow]")
            return
        show_welcome(profile)

    while True:
        try:
            show_dashboard(profile)
            choice = Prompt.ask(
                "[bold bright_cyan]SELECT[/bold bright_cyan]",
                choices=["0", "1", "2", "3", "4", "5", "6", "7", "8"],
                show_choices=False,
                default="1",
            )
            if choice == "1":
                run_module("main.py", "RECORD SPEECH")
            elif choice == "2":
                run_module("transcript.py", "TRANSCRIBE SPEECH")
            elif choice == "3":
                run_module("speech_analyzer.py", "ANALYZE SPEECH")
            elif choice == "4":
                browse_files(list_recordings(), "MY RECORDINGS",
                             "No supported audio recordings found.")
            elif choice == "5":
                browse_files(list_transcripts(), "MY TRANSCRIPTS",
                             "No saved transcripts found.", open_as_text=True)
            elif choice == "6":
                browse_files(list_reports(), "ANALYSIS REPORTS",
                             "No analysis reports found.", open_as_text=True)
            elif choice == "7":
                profile = manage_profile(profile)
            elif choice == "8":
                run_module("practice_engine.py", "PERSONAL PRACTICE")
            elif choice == "0":
                clear()
                console.print(Panel(
                    Align.center(Text.assemble(
                        ("THANK YOU FOR USING\n", "white"),
                        ("ELOQUIFY\n", "bold bright_cyan"),
                        ("Keep speaking. Keep improving.", "bold bright_magenta"),
                    )),
                    border_style="bright_magenta", box=box.DOUBLE, padding=(2, 1),
                ))
                break
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Goodbye![/yellow]")
            break


if __name__ == "__main__":
    main()
