from pathlib import Path

from annotile.dataloader.dataloader import (
    Dataloader,
    get_yolo_annotation_paths,
    get_yolo_image_paths,
)
from annotile.image_tiler.image_tiler import ImageTiler
from annotile.label_tiler.label_tiler import LabelTiler


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

        # TODO: don't make num_tiles mandatory, compute from image size and tile size
        # Vice versa can also be done, need to consider this
        # Below is just a placeholder

        # Initialize the image tiler
        self.image_tiler = ImageTiler(overlap_pct, tile_size, (0, 0))

        # Initialize the label tiler
        self.label_tiler = LabelTiler(overlap_pct, tile_size, (0, 0))

    def tile_dataset(self, save_image_dir: Path, save_label_dir: Path):
        """Tiles the entire dataset of images and labels.

        Args:
            save_image_dir (Path): Directory to save tiled images.
            save_label_dir (Path): Directory to save tiled labels.
        """
        for image_path, annotation_path in self.dataloader.paired:
            # Tile the image
            tiles = self.image_tiler.tile_image(image_path=image_path)

            # Tile the labels
            tiles_to_labels = self.label_tiler.tile_labels(annotation_path=annotation_path, tiles=tiles)

            # Save the tiled images
            self.image_tiler.save_tiles(tiles_to_save=tiles, save_dir=save_image_dir)

            # Save the tiled labels
            self.label_tiler.save_tiles(tiles_to_labels=tiles_to_labels, save_dir=save_label_dir)
