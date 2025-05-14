from pathlib import Path

from annotile.dataloader.dataloader import (
    Dataloader,
    get_yolo_annotation_paths,
    get_yolo_image_paths,
)
from annotile.image_tiler.image_tiler import ImageTiler


class Tiler:
    def __init__(
        self,
        image_dir: Path,
        annotation_dir: Path,
        tile_size: tuple[int, int],
        overlap_pct: float,
    ):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.tile_size = tile_size
        self.overlap_pct = overlap_pct

        # Initialize the dataloader
        image_paths = get_yolo_image_paths(image_dir)
        annotation_paths = get_yolo_annotation_paths(annotation_dir)
        self.dataloader = Dataloader(image_paths=image_paths, annotation_paths=annotation_paths)

        # Initialize the image tiler
        self.image_tiler = ImageTiler(overlap_pct, tile_size, (0, 0))
