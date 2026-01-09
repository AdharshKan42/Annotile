"""DTO models for dataloader module."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataloaderResult:
    """Result from dataloader containing paired and unpaired files."""

    paired: list[tuple[Path, Path]]
    unmatched_images: list[Path]
    unmatched_annotations: list[Path]
