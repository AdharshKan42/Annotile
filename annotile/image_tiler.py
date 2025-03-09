from pathlib import Path
from typing import Tuple
import numpy as np
from PIL import Image

class ImageTiler:
    def __init__(self, overlap, tile_size, image_size, num_tiles):
        self.overlap = overlap
        self.tile_size = tile_size
        self.image_size = image_size
        self.num_tiles = num_tiles


    def num_tiles_to_tile_sizes(self, num_tiles: Tuple[int, int], overlap: float) -> Tuple[int, int]:
        """
        Convert number of tiles to tile sizes.

        Args:
            num_tiles (Tuple[int, int]): Number of tiles in (width, height).

        Returns:
            Tuple[int, int]: Tile sizes in (tile_width, tile_height).
        """
        tile_width = self.image_size[0] // num_tiles[0]
        tile_height = self.image_size[1] // num_tiles[1]
        return int(tile_width * (1 + overlap)), int(tile_height * (1 + overlap))

    def tile_image(self, image_path: Path, tile_size: Tuple[int, int], overlap: float, num_tiles: Tuple[int,int], og_tile_size: Tuple[int,int]) -> np.ndarray:
        """
        Splits an image into overlapping tiles using vectorized approach.

        Args:
            image_path (Path): Path to the image.
            tile_size (Tuple[int, int]): (tile_width, tile_height).
            overlap (float): Overlap fraction (0.0 to 1.0).

        Returns:
            np.ndarray: Array of tiles (shape: (num_tiles, tile_height, tile_width, channels)).
        """
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
        tiles = tiled.reshape(num_tiles_y * num_tiles_x, tile_height, tile_width, image.shape[2])

        return tiles

    def save_tiles(self, tiles: np.ndarray, save_dir: Path):
        """
        Save the tiles to the specified directory.

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

    def process_image(self, image_path: Path, save_dir: Path,  num_tiles: Tuple[int,int], og_tile_size: Tuple[int,int]):
        """
        Process the image: tile it and save the tiles.

        Args:
            image_path (Path): Path to the image.
            save_dir (Path): Directory to save the tiles.
        """
        tiles = self.tile_image(image_path, self.tile_size, self.overlap, num_tiles, og_tile_size)
        self.save_tiles(tiles, save_dir)

    def save_metadata(self):
        """
        Save metadata about the tiling process.
        """
        
        with open("metadata.txt", "w") as f:
            f.write(f"Tile Size: {self.tile_size}\n")
            f.write(f"Overlap Percentage: {self.overlap}\n")
            f.write(f"Image Size: {self.image_size}\n")

