from pathlib import Path
import logging
import math
from PIL import Image
from annotile.dataloader.dataloader import Dataloader, get_yolo_annotation_paths, get_yolo_image_paths
from annotile.image_tiler.image_tiler import ImageTiler
from annotile.label_tiler.label_tiler import LabelTiler
from multiprocessing import Pool
import os


class Tiler:
    def __init__(
        self,
        image_dir: Path,
        annotation_dir: Path,
        tile_size: tuple[int, int] = None,
        overlap: float = 0.0,
        num_tiles: tuple[int, int] = None,
    ):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.tile_size = tile_size
        self.overlap = overlap
        self.num_tiles = num_tiles

        image_paths = get_yolo_image_paths(image_dir)
        annotation_paths = get_yolo_annotation_paths(annotation_dir)
        self.dataloader = Dataloader(image_paths=image_paths, annotation_paths=annotation_paths)

    def _compute_num_tiles(self, image_size, tile_size, overlap):
        step_x = int(tile_size[0] * (1 - overlap))
        step_y = int(tile_size[1] * (1 - overlap))
        if step_x <= 0 or step_y <= 0:
            raise ValueError("Invalid tile size or overlap percentage.")
        num_tiles_x = math.ceil((image_size[0] - tile_size[0]) / step_x) + 1
        num_tiles_y = math.ceil((image_size[1] - tile_size[1]) / step_y) + 1
        return (num_tiles_x, num_tiles_y)

    def _compute_tile_size(self, image_size, num_tiles, overlap):
        if num_tiles[0] <= 0 or num_tiles[1] <= 0:
            raise ValueError("num_tiles must be positive.")
        step_x = image_size[0] / (num_tiles[0] - 1) if num_tiles[0] > 1 else image_size[0]
        step_y = image_size[1] / (num_tiles[1] - 1) if num_tiles[1] > 1 else image_size[1]
        tile_width = int(step_x / (1 - overlap))
        tile_height = int(step_y / (1 - overlap))
        return (tile_width, tile_height)

    def tile_dataset(self, save_image_dir: Path, save_label_dir: Path):
        """Tiles the entire dataset of images and labels.

        Args:
            save_image_dir (Path): Directory to save tiled images.
            save_label_dir (Path): Directory to save tiled labels.
        """
        for image_path, annotation_path in self.dataloader.paired:
            with Image.open(image_path) as img:
                image_size = img.size

            # Compute missing parameters for each image
            tile_size = self.tile_size
            num_tiles = self.num_tiles

            # Determine tile_size and num_tiles per image based on provided parameters
            # TODO: refactor this logic into its own method
            # TODO: have support to tile one image and its annotation at a time, so multiprocessing can be used
            if tile_size and not num_tiles:
                num_tiles = self._compute_num_tiles(image_size, tile_size, self.overlap)
            elif num_tiles and not tile_size:
                tile_size = self._compute_tile_size(image_size, num_tiles, self.overlap)
            elif tile_size and num_tiles:
                expected_num_tiles = self._compute_num_tiles(image_size, tile_size, self.overlap)
                if expected_num_tiles != num_tiles:
                    logging.error(
                        f"Requested num_tiles {num_tiles} not possible with image size {image_size} and tile size {tile_size}."
                    )
                    raise ValueError("Requested num_tiles not possible with current image and tile size.")
            else:
                raise ValueError("Either tile_size or num_tiles must be provided.")

            image_tiler = ImageTiler(
                overlap=self.overlap,
                tile_size=tile_size,
                image_size=image_size,
                num_tiles=num_tiles,
                image_path=image_path,
            )
            label_tiler = LabelTiler(
                overlap=self.overlap,
                tile_size=tile_size,
                image_size=image_size,
                num_tiles=num_tiles,
                label_path=annotation_path,
            )

            tiles = image_tiler.tile_image(image_path=image_path)
            tiles_to_labels = label_tiler.tile_labels(label_path=annotation_path, tiles=tiles)
            image_tiler.save_tiles(tiles_to_save=tiles, save_dir=save_image_dir)
            label_tiler.save_tiles(tiles_to_labels=tiles_to_labels, save_dir=save_label_dir)

    def tile_image_and_label(
        self,
        image_path: Path,
        annotation_path: Path,
        save_image_dir: Path,
        save_label_dir: Path,
    ):
        """Tiles a single image and its corresponding annotation.

        Args:
            image_path (Path): Path to the image to be tiled.
            annotation_path (Path): Path to the corresponding annotation.
            save_image_dir (Path): Directory to save tiled images.
            save_label_dir (Path): Directory to save tiled labels.
        """
        with Image.open(image_path) as img:
            image_size = img.size

        tile_size = self.tile_size
        num_tiles = self.num_tiles

        if tile_size and not num_tiles:
            num_tiles = self._compute_num_tiles(image_size, tile_size, self.overlap)
        elif num_tiles and not tile_size:
            tile_size = self._compute_tile_size(image_size, num_tiles, self.overlap)
        elif tile_size and num_tiles:
            expected_num_tiles = self._compute_num_tiles(image_size, tile_size, self.overlap)
            if expected_num_tiles != num_tiles:
                logging.error(
                    f"Requested num_tiles {num_tiles} not possible with image size {image_size} and tile size {tile_size}."
                )
                raise ValueError("Requested num_tiles not possible with current image and tile size.")
        else:
            raise ValueError("Either tile_size or num_tiles must be provided.")

        image_tiler = ImageTiler(
            overlap=self.overlap,
            tile_size=tile_size,
            image_size=image_size,
            num_tiles=num_tiles,
            image_path=image_path,
        )
        label_tiler = LabelTiler(
            overlap=self.overlap,
            tile_size=tile_size,
            image_size=image_size,
            num_tiles=num_tiles,
            label_path=annotation_path,
        )

        tiles = image_tiler.tile_image(image_path=image_path)
        tiles_to_labels = label_tiler.tile_labels(label_path=annotation_path, tiles=tiles)
        image_tiler.save_tiles(tiles_to_save=tiles, save_dir=save_image_dir)
        label_tiler.save_tiles(tiles_to_labels=tiles_to_labels, save_dir=save_label_dir)

    def multiprocess_tile_dataset(self, save_image_dir: Path, save_label_dir: Path, num_workers: int | None = None):
        """Tiles the entire dataset of images and labels using multiprocessing.

        Args:
            save_image_dir (Path): Directory to save tiled images.
            save_label_dir (Path): Directory to save tiled labels.
            num_workers (int | None): Number of worker processes to use.
        """

        if num_workers is None:
            num_workers = os.cpu_count()

        def worker(args):
            image_path, annotation_path = args
            self.tile_image_and_label(
                image_path=image_path,
                annotation_path=annotation_path,
                save_image_dir=save_image_dir,
                save_label_dir=save_label_dir,
            )

        with Pool(processes=num_workers) as pool:
            pool.map(worker, self.dataloader.paired)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Example usage
    image_dir = Path("/Users/adharshkandula/Documents/Repos/Annotile/tests/data/test_img")
    tiler = Tiler(
        image_dir=image_dir,
        annotation_dir=image_dir.parent / "test_labels",
        tile_size=(512, 512),
        overlap=0.2,
    )
    tiler.tile_dataset(
        save_image_dir=image_dir.parent / "tiled_images",
        save_label_dir=image_dir.parent / "tiled_labels",
    )
