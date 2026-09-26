"""Small dependency-free progress indicator for long competition runs."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import TextIO


def _duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m"
    if minutes:
        return f"{minutes:d}m{seconds:02d}s"
    return f"{seconds:d}s"


@dataclass
class ProgressBar:
    """Print compact progress snapshots without adding a runtime dependency."""

    description: str
    total: int
    enabled: bool = True
    width: int = 24
    stream: TextIO = field(default_factory=lambda: sys.stderr)
    current: int = field(default=0, init=False)
    started: float = field(default_factory=time.perf_counter, init=False)

    def __post_init__(self) -> None:
        if self.total < 1:
            raise ValueError("progress total must be positive")
        if self.width < 5:
            raise ValueError("progress width must be at least 5")

    def start(self) -> None:
        self._render()

    def update(self, steps: int = 1) -> None:
        if steps < 0:
            raise ValueError("progress steps cannot be negative")
        self.current = min(self.total, self.current + steps)
        self._render()

    def _render(self) -> None:
        if not self.enabled:
            return
        elapsed = time.perf_counter() - self.started
        fraction = self.current / self.total
        filled = min(self.width, int(self.width * fraction))
        bar = "#" * filled + "-" * (self.width - filled)
        eta = elapsed * (self.total - self.current) / self.current if self.current else 0.0
        eta_text = _duration(eta) if self.current < self.total and self.current else "--"
        print(
            f"{self.description} [{bar}] {self.current}/{self.total} "
            f"({fraction:6.1%}) elapsed={_duration(elapsed)} eta={eta_text}",
            file=self.stream,
            flush=True,
        )
