"""Managed Phase-2 output paths with traversal-safe experiment identifiers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


@dataclass(frozen=True, slots=True)
class Phase2Paths:
    experiment_id: str
    output_root: Path = Path("outputs")

    def __post_init__(self) -> None:
        if not _SAFE_ID.fullmatch(self.experiment_id):
            raise ValueError(f"experiment_id must match {_SAFE_ID.pattern}")

    def experiment_root(self) -> Path:
        return self.output_root / self.experiment_id

    def reflection_dir(self, domain: str) -> Path:
        return self.experiment_root() / "build" / domain / "raw_reflections"

    def reflection_path(self, domain: str, task_id: str) -> Path:
        return self.reflection_dir(domain) / f"{task_id}.json"

    def artifact_path(self, domain: str) -> Path:
        return self.experiment_root() / "artifacts" / domain / "frozen_lessons.json"

    def publication_path(self, domain: str) -> Path:
        return self.experiment_root() / "publication" / f"{domain}.json"
