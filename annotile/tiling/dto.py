"""Top-level DTO models for the Tiler class."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from pydantic import BaseModel, ConfigDict, model_validator


class TilerConfig(BaseModel):
    """Configuration for the Tiler class.

    Either `tile_size` or `num_tiles` must be provided.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    image_dir: Path
    annotation_dir: Optional[Path] = None

    save_image_dir: Path
    save_label_dir: Optional[Path] = None

    overlap: float = 0.2
    tile_size: Optional[Tuple[int, int]] = None
    num_tiles: Optional[Tuple[int, int]] = None
    og_tile_size: Optional[Tuple[int, int]] = None

    use_paired: bool = True  # paired images+labels vs images-only

    @model_validator(mode="after")
    def _validate_mutual_requirements(self) -> "TilerConfig":
        if self.use_paired:
            if self.annotation_dir is None:
                raise ValueError("When use_paired=True, annotation_dir must be provided.")
            if self.save_label_dir is None:
                raise ValueError("When use_paired=True, save_label_dir must be provided.")
        if self.tile_size is None and self.num_tiles is None:
            raise ValueError("Either tile_size or num_tiles must be provided.")
        return self
