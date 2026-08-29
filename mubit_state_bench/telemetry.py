"""Separate append-only telemetry for Mubit retrieval calls."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

_WRITE_LOCK = threading.Lock()


class JsonlTelemetrySink:
    """Write one retrieval event per line, outside STATE-Bench trajectories."""

    def __init__(self, path: Path):
        self.path = path

    def write(self, event: dict[str, Any]) -> None:
        payload = json.dumps(event, ensure_ascii=False, sort_keys=True, default=str)
        with _WRITE_LOCK:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(payload + "\n")
