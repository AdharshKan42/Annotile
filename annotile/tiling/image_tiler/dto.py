"""DTO models for ImageTiler module."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from pydantic import BaseModel, ConfigDict


class ImageTilerConfig(BaseModel):
    """Configuration DTO for ImageTiler class.
    
    Passed to ImageTiler for a single image tiling operation.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    image_path: Path
    image_size: Tuple[int, int]

    overlap: float
    og_tile_size: Tuple[int, int] = (512, 512)
    tile_size: Optional[Tuple[int, int]] = None
    num_tiles: Optional[Tuple[int, int]] = None

    save_dir: Path
