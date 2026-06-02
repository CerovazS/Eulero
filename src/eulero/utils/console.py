# =============================================================================
# Lightweight console helpers (vendored, minimal).
# Uses ``rich`` if available, otherwise falls back to plain printing.
# =============================================================================
import os as _os

try:  # rich is optional; degrade gracefully if absent
    from rich.console import Console

    _console = Console(no_color=bool(_os.environ.get("NO_COLOR")))

    def _emit(msg: str, prefix: str, style: str) -> None:
        prefix_str = f"[[bold cyan]{prefix}[/bold cyan]] " if prefix else ""
        _console.print(f"{prefix_str}{msg}", style=style)
except Exception:  # pragma: no cover - exercised only when rich is missing
    def _emit(msg: str, prefix: str, style: str) -> None:
        prefix_str = f"[{prefix}] " if prefix else ""
        print(f"{prefix_str}{msg}")


def ok(msg: str, prefix: str = "") -> None:
    _emit(msg, prefix, "bold green")


def warn(msg: str, prefix: str = "") -> None:
    _emit(msg, prefix, "bold yellow")


def err(msg: str, prefix: str = "") -> None:
    _emit(msg, prefix, "bold red")


def info(msg: str, prefix: str = "") -> None:
    _emit(msg, prefix, "cyan")
