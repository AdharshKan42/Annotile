import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
from pydantic import BaseModel, model_validator
from shapely import Polygon


class Tile(BaseModel):
    image_array: np.ndarray
    position: tuple[int, int]
    # Top left coords of tile relative to image
    position_in_image: tuple[int, int]
    tile_size: tuple[int, int]
    image_size: tuple[int, int]
    overlap: float
    image_path: Path | None = None
    polygon: Polygon | None = None

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def create_polygon(self) -> "Tile":
        self.polygon = Polygon(
            [
                (self.position_in_image[0], self.position_in_image[1]),
                (self.position_in_image[0], self.position_in_image[1] + self.tile_size[1]),
                (self.position_in_image[0] + self.tile_size[0], self.position_in_image[1]),
                (self.position_in_image[0] + self.tile_size[0], self.position_in_image[1] + self.tile_size[1]),
            ]
        )
        return self

    def _image_digest(self) -> int:
        """Return a small integer digest for the image content (fast to compare & include in hash)."""
        # ensure contiguous bytes
        arr = np.ascontiguousarray(self.image_array)
        # use first 8 bytes of sha256 for a compact stable int
        digest = hashlib.sha256(arr.tobytes()).digest()[:8]
        return int.from_bytes(digest, "big")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Tile):
            return NotImplemented
        return (
            self.position == other.position
            and self.position_in_image == other.position_in_image
            and self.tile_size == other.tile_size
            and self.image_size == other.image_size
            and float(self.overlap) == float(other.overlap)
            and self._image_digest() == other._image_digest()
        )

    def __hash__(self) -> int:
        # combine metadata and image digest into a stable hashable tuple
        return hash(
            (
                self.position,
                self.position_in_image,
                self.tile_size,
                self.image_size,
                round(float(self.overlap), 6),
                self._image_digest(),
            )
        )


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
                img_arr = tiled[y, x, :, :, :]
                pos = (y, x)
                pos_in_image = (y * step_y, x * step_x)
                t = Tile(
                    image_array=img_arr,
                    position=pos,
                    position_in_image=pos_in_image,
                    tile_size=tile_size,
                    image_size=self.image_size,
                    overlap=overlap,
                    image_path=image_path,
                )
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
