from pathlib import Path

import numpy as np
from PIL import Image
from typing import Self
from pydantic import BaseModel, model_validator
from shapely import Polygon


class Tile(BaseModel):
    image_array: np.ndarray
    position: tuple[int]
    # Top left coords of tile relative to image
    position_in_image: tuple[int]
    tile_size: tuple[int]
    image_size: tuple[int]
    overlap: float
    polygon: Polygon | None = None

    @model_validator(mode="after")
    def create_polygon(self) -> Self:
        self.polygon = Polygon([(self.position_in_image[0], self.position_in_image[1]), (self.position_in_image[0], self.position_in_image[1] + self.tile_size[1]),  (self.position_in_image[0] + self.tile_size[0], self.position_in_image[1]),
                                (self.position_in_image[0] + self.tile_size[0], self.position_in_image[1] + self.tile_size[1])])
        return self


class ImageTiler:
    def __init__(
        self,
        overlap,
        tile_size,
        image_size,
        num_tiles,
        image_path=None,
        save_dir=None,
        og_tile_size=None,
    ):
        self.overlap = overlap
        self.tile_size = tile_size
        self.image_size = image_size
        self.num_tiles = num_tiles
        self.image_path = image_path
        self.save_dir = save_dir
        self.og_tile_size = og_tile_size

    def num_tiles_to_tile_sizes(
        self, num_tiles: tuple[int, int] | None = None, overlap: float | None = None
    ) -> tuple[int, int]:
        """Convert number of tiles to tile sizes.

        Args:
            num_tiles (Tuple[int, int] | None): Number of tiles in (width, height).
            overlap (float | None): Percent of overlap between tiles from 0.0 to 1.0.


        Returns:
            Tuple[int, int]: Tile sizes in (tile_width, tile_height).
        """
        num_tiles = num_tiles or self.num_tiles
        overlap = overlap or self.overlap

        tile_width = self.image_size[0] // num_tiles[0]
        tile_height = self.image_size[1] // num_tiles[1]
        return int(tile_width * (1 + overlap)), int(tile_height * (1 + overlap))

    def tile_image(
        self,
        image_path: Path | None = None,
        tile_size: tuple[int, int] | None = None,
        overlap: float | None = None,
        num_tiles: tuple[int, int] | None = None,
        og_tile_size: tuple[int, int] | None = None,
    ) -> list[Tile]:
        """Splits an image into overlapping tiles using vectorized approach.

        Args:
            image_path (Path | None): Path to the image.
            tile_size (Tuple[int, int] | None): (tile_width, tile_height).
            overlap (float | None): Percent of overlap between tiles from 0.0 to 1.0.
            num_tiles (tuple[int, int] | None): Number of tiles in (width, height).
            og_tile_size (tuple[int, int] | None): Original (tile_width, tile_height).

        Returns:
            np.ndarray: Array of tiles (shape: (num_tiles, tile_height, tile_width, channels)).
        """
        image_path = image_path or self.image_path
        tile_size = tile_size or self.tile_size
        overlap = overlap or self.overlap
        num_tiles = num_tiles or self.num_tiles
        og_tile_size = og_tile_size or self.og_tile_size

        tile_size = self.num_tiles_to_tile_sizes(num_tiles, overlap)

        # Load the image using PIL
        image = np.array(Image.open(image_path))
        img_height, img_width = image.shape[:2]
        tile_height, tile_width = tile_size

        # Calculate step size
        og_tile_height, og_tile_width = og_tile_size
        step_y = int(og_tile_height * (1 - overlap))
        step_x = int(og_tile_width * (1 - overlap))

        # Ensure the image is large enough
        if img_height < tile_height or img_width < tile_width:
            raise ValueError("Tile size is larger than the image dimensions.")

        # Create sliding windows of shape (tile_height, tile_width, channels)
        windows = np.lib.stride_tricks.sliding_window_view(image, (tile_height, tile_width, image.shape[2]))

        # Apply step to subsample the sliding windows
        tiled = windows[::step_y, ::step_x, 0, :, :, :]

        # Reshape to flat array of tiles
        num_tiles_y, num_tiles_x = tiled.shape[:2]
        output_tiles = []

        for y in range(num_tiles_y):
            for x in range(num_tiles_x):
                t = Tile()
                t.image_array = tiled[y, x, :, :, :]
                t.position = (y, x)
                t.position_xy = (y * step_y, x * step_x)
                t.tile_size = tile_size
                t.image_size = self.image_size
                t.overlap = overlap
                output_tiles.append(t)

        return output_tiles

    def save_tiles(self, tiles: np.ndarray, save_dir: Path) -> None:
        """Save the tiles to the specified directory.

        Args:
            tiles (np.ndarray): Array of tiles.
            save_dir (Path): Directory to save the tiles.
        """
        if not save_dir.exists():
            save_dir.mkdir(parents=True, exist_ok=True)

        for i, tile in enumerate(tiles):
            tile_path = save_dir / f"tile_{i}.png"
            # Save the tile using PIL
            Image.fromarray(tile).save(tile_path)

    def process_image(
        self,
        image_path: Path | None = None,
        save_dir: Path | None = None,
        num_tiles: tuple[int, int] | None = None,
        og_tile_size: tuple[int, int] | None = None,
    ):
        """Process the image: tile it and save the tiles.

        Args:
            image_path (Path | None): Path to the image.
            save_dir (Path | None): Directory to save the tiles.
            num_tiles (tuple[int, int] | None): Number of tiles in (width, height).
            og_tile_size (tuple[int, int] | None): Original (tile_width, tile_height).

        """
        image_path = image_path or self.image_path
        save_dir = save_dir or self.save_dir
        num_tiles = num_tiles or self.num_tiles
        og_tile_size = og_tile_size or self.og_tile_size

        tiles = self.tile_image(image_path, self.tile_size, self.overlap, num_tiles, og_tile_size)
        self.save_tiles(tiles, save_dir)

    def save_metadata(self):
        """Save metadata about the tiling process."""
        with open("metadata.txt", "w") as f:
            f.write(f"Tile Size: {self.tile_size}\n")
            f.write(f"Overlap Percentage: {self.overlap}\n")
            f.write(f"Image Size: {self.image_size}\n")
