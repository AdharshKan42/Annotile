"""Tiler module for coordinating image and label tiling operations."""

import logging
import os
from multiprocessing import Pool
from pathlib import Path

from PIL import Image

from annotile.dataloader.dataloader import load_dataset
from annotile.tiling.dto import TilerConfig
from annotile.tiling.image_tiler.dto import ImageTilerConfig
from annotile.tiling.image_tiler.image_tiler import ImageTiler, Tile
from annotile.tiling.label_tiler.label_tiler import LabelTiler


class Tiler:
    """Main tiler class that coordinates image and label tiling.
    
    Uses configuration DTOs to pass parameters to tilers.
    """

    def __init__(self, config: TilerConfig):
        """Initialize Tiler with configuration DTO.

        Args:
            config: TilerConfig containing all tiling parameters.
        """
        self.config = config
        self.dataset = load_dataset(config.image_dir, config.annotation_dir)

    def _tile_single_image(self, image_path: Path) -> list[Tile]:
        """Tile a single image.
        
        Args:
            image_path: Path to the image file.
            
        Returns:
            List of Tile objects.
        """
        with Image.open(image_path) as img:
            image_size = img.size

        img_config = ImageTilerConfig(
            image_path=image_path,
            image_size=image_size,
            overlap=self.config.overlap,
            og_tile_size=self.config.og_tile_size,
            tile_size=self.config.tile_size,
            num_tiles=self.config.num_tiles,
            save_dir=self.config.save_image_dir,
        )

        image_tiler = ImageTiler(img_config)
        return image_tiler.process()

    def _tile_paired(self, image_path: Path, label_path: Path) -> tuple[list[Tile], dict]:
        """Tile an image and its corresponding labels.
        
        Args:
            image_path: Path to the image file.
            label_path: Path to the label file.
            
        Returns:
            Tuple of (tiles, tiles_to_labels dict).
        """
        with Image.open(image_path) as img:
            image_size = img.size

        # Create and process image tiler
        img_config = ImageTilerConfig(
            image_path=image_path,
            image_size=image_size,
            overlap=self.config.overlap,
            og_tile_size=self.config.og_tile_size,
            tile_size=self.config.tile_size,
            num_tiles=self.config.num_tiles,
            save_dir=self.config.save_image_dir,
        )
        image_tiler = ImageTiler(img_config)
        tiles = image_tiler.process()

        # Create and process label tiler
        label_config = image_tiler.get_label_tiler_config(
            label_path=label_path,
            save_dir=self.config.save_label_dir,  # type: ignore[arg-type]
        )
        label_tiler = LabelTiler(label_config)
        tiles_to_labels = label_tiler.process(tiles)

        return tiles, tiles_to_labels

    def tile_dataset(self) -> None:
        """Tile the entire dataset (paired or images-only)."""
        if self.config.use_paired:
            for image_path, label_path in self.dataset.paired:
                self._tile_paired(image_path, label_path)
                logging.info(f"Tiled paired: {image_path.name}")
        else:
            for image_path in self.dataset.unmatched_images:
                self._tile_single_image(image_path)
                logging.info(f"Tiled image: {image_path.name}")

    def multiprocess_tile_dataset(self, num_workers: int | None = None) -> None:
        """Tile the entire dataset using multiprocessing.

        Args:
            num_workers: Number of worker processes to use. Defaults to CPU count.
        """
        if num_workers is None:
            num_workers = os.cpu_count()

        if self.config.use_paired:
            worker_fn = _tile_paired_worker
            tasks = [
                (image_path, label_path, self.config)
                for image_path, label_path in self.dataset.paired
            ]
        else:
            worker_fn = _tile_image_worker
            tasks = [(image_path, self.config) for image_path in self.dataset.unmatched_images]

        with Pool(processes=num_workers) as pool:
            pool.map(worker_fn, tasks)


def _tile_image_worker(args: tuple[Path, TilerConfig]) -> None:
    """Worker function to tile a single image.
    
    Args:
        args: Tuple of (image_path, config).
    """
    image_path, config = args
    
    with Image.open(image_path) as img:
        image_size = img.size

    img_config = ImageTilerConfig(
        image_path=image_path,
        image_size=image_size,
        overlap=config.overlap,
        og_tile_size=config.og_tile_size,
        tile_size=config.tile_size,
        num_tiles=config.num_tiles,
        save_dir=config.save_image_dir,
    )

    image_tiler = ImageTiler(img_config)
    image_tiler.process()


def _tile_paired_worker(args: tuple[Path, Path, TilerConfig]) -> None:
    """Worker function to tile an image and its labels.
    
    Args:
        args: Tuple of (image_path, label_path, config).
    """
    image_path, label_path, config = args
    
    with Image.open(image_path) as img:
        image_size = img.size

    # Create and process image tiler
    img_config = ImageTilerConfig(
        image_path=image_path,
        image_size=image_size,
        overlap=config.overlap,
        og_tile_size=config.og_tile_size,
        tile_size=config.tile_size,
        num_tiles=config.num_tiles,
        save_dir=config.save_image_dir,
    )
    image_tiler = ImageTiler(img_config)
    tiles = image_tiler.process()

    # Create and process label tiler
    label_config = image_tiler.get_label_tiler_config(
        label_path=label_path,
        save_dir=config.save_label_dir,  # type: ignore[arg-type]
    )
    label_tiler = LabelTiler(label_config)
    label_tiler.process(tiles)
