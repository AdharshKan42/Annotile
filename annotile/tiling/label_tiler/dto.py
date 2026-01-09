from dataclasses import dataclass
from pathlib import Path


@dataclass
class LabelTilerConfig:
    """Configuration for label tiling (all fields required and validated by ImageTiler).
    
    This config is created by ImageTiler.get_label_tiler_config() after validation.
    All fields are guaranteed to be valid when this config is instantiated.
    """
    label_path: Path
    image_size: tuple[int, int]
    tile_size: tuple[int, int]
    overlap: float
    save_dir: Path
