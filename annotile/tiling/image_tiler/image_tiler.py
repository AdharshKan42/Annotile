import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
from pydantic import BaseModel, model_validator
from shapely import Polygon

from annotile.tiling.image_tiler.dto import ImageTilerConfig
from annotile.tiling.label_tiler.dto import LabelTilerConfig


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
    """Tiles images using configuration DTO.
    
    Validates and computes missing parameters (tile_size or num_tiles).
    """

    def __init__(self, config: ImageTilerConfig):
        """Initialize ImageTiler with configuration DTO.

        Args:
            config: ImageTilerConfig containing all required parameters.
        """
        self.config = config
        self._validate_and_compute_parameters()

    def _validate_and_compute_parameters(self):
        """Validate config and compute missing tile_size or num_tiles."""
        if self.config.tile_size is None and self.config.num_tiles is None:
            raise ValueError("Either tile_size or num_tiles must be provided in config.")

        if self.config.tile_size and not self.config.num_tiles:
            self.config.num_tiles = self._compute_num_tiles(
                self.config.image_size, self.config.tile_size, self.config.overlap
            )
        elif self.config.num_tiles and not self.config.tile_size:
            self.config.tile_size = self._compute_tile_size(
                self.config.image_size, self.config.num_tiles, self.config.overlap
            )
        elif self.config.tile_size and self.config.num_tiles:
            expected = self._compute_num_tiles(
                self.config.image_size, self.config.tile_size, self.config.overlap
            )
            if expected != self.config.num_tiles:
                raise ValueError(
                    f"Requested num_tiles {self.config.num_tiles} not possible with "
                    f"image_size {self.config.image_size} and tile_size {self.config.tile_size}."
                )

    @staticmethod
    def _compute_num_tiles(
        image_size: tuple[int, int], tile_size: tuple[int, int], overlap: float
    ) -> tuple[int, int]:
        """Compute number of tiles from image size, tile size, and overlap."""
        step_x = int(tile_size[0] * (1 - overlap))
        step_y = int(tile_size[1] * (1 - overlap))
        if step_x <= 0 or step_y <= 0:
            raise ValueError("Invalid tile size or overlap percentage.")
        num_tiles_x = ((image_size[0] - tile_size[0] + step_x - 1) // step_x) + 1
        num_tiles_y = ((image_size[1] - tile_size[1] + step_y - 1) // step_y) + 1
        return (num_tiles_x, num_tiles_y)

    @staticmethod
    def _compute_tile_size(
        image_size: tuple[int, int], num_tiles: tuple[int, int], overlap: float
    ) -> tuple[int, int]:
        """Compute tile size from image size, number of tiles, and overlap."""
        if num_tiles[0] <= 0 or num_tiles[1] <= 0:
            raise ValueError("num_tiles must be positive.")
        step_x = image_size[0] / (num_tiles[0] - 1) if num_tiles[0] > 1 else image_size[0]
        step_y = image_size[1] / (num_tiles[1] - 1) if num_tiles[1] > 1 else image_size[1]
        tile_width = int(step_x / (1 - overlap))
        tile_height = int(step_y / (1 - overlap))
        return (tile_width, tile_height)

    def get_label_tiler_config(self, label_path: Path, save_dir: Path) -> LabelTilerConfig:
        """Create LabelTilerConfig from validated ImageTiler parameters.
        
        Args:
            label_path: Path to the label file.
            save_dir: Directory to save tiled labels.
            
        Returns:
            LabelTilerConfig with all computed parameters.
        """
        if self.config.tile_size is None or self.config.num_tiles is None:
            raise ValueError("tile_size and num_tiles must be computed before creating LabelTilerConfig.")
        
        return LabelTilerConfig(
            label_path=label_path,
            overlap=self.config.overlap,
            tile_size=self.config.tile_size,
            image_size=self.config.image_size,
            num_tiles=self.config.num_tiles,
            og_tile_size=self.config.og_tile_size,
            save_dir=save_dir,
        )

    def tile_image(self) -> list[Tile]:
        """Splits an image into overlapping tiles using vectorized approach.

        Returns:
            list[Tile]: List of Tile objects containing image arrays and metadata.
        """
        # Load the image using PIL
        image = np.array(Image.open(self.config.image_path))
        img_height, img_width = image.shape[:2]
        
        # Tile size must be computed at this point
        assert self.config.tile_size is not None, "tile_size should be computed in __init__"
        assert self.config.num_tiles is not None, "num_tiles should be computed in __init__"
        
        tile_height, tile_width = self.config.tile_size

        # Calculate step size
        og_tile_height, og_tile_width = self.config.og_tile_size
        step_y = int(og_tile_height * (1 - self.config.overlap))
        step_x = int(og_tile_width * (1 - self.config.overlap))

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
                    tile_size=self.config.tile_size,
                    image_size=self.config.image_size,
                    overlap=self.config.overlap,
                    image_path=self.config.image_path,
                )
                output_tiles.append(t)

        return output_tiles

    def save_tiles(self, tiles: list[Tile]) -> None:
        """Save the tiles to the specified directory.

        Args:
            tiles: List of Tile objects to save.
        """
        save_dir = self.config.save_dir
        if not save_dir.exists():
            save_dir.mkdir(parents=True, exist_ok=True)

        for tile in tiles:
            tile_path = save_dir / f"{tile.image_path.stem}_{tile.position_in_image}.png"
            # Save the tile using PIL
            Image.fromarray(tile.image_array).save(tile_path)

    def process(self) -> list[Tile]:
        """Process the image: tile it and save the tiles.

        Returns:
            list[Tile]: List of generated tiles.
        """
        tiles = self.tile_image()
        self.save_tiles(tiles)
        return tiles
