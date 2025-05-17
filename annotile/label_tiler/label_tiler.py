from pathlib import Path

from pydantic import BaseModel, model_validator
from typing import Self

from annotile.image_tiler.image_tiler import Tile
from shapely import Polygon


class Label(BaseModel):
    object_class: int
    x: int
    y: int
    width: int
    height: int
    image_width: int
    image_height: int

    # Filled after validation
    x_1: int = 0
    y_1: int = 0
    x_2: int = 0
    y_2: int = 0
    polygon: Polygon | None = None

    @model_validator(mode="after")
    def convert_xywh_to_bbox(self) -> Self:
        self.x_1 = int((self.x - self.width/2) * self.image_width)
        self.y_1 = int((self.y - self.height/2) * self.image_height)
        self.x_2 = int((self.x + self.width/2) * self.image_width)
        self.y_2 = int((self.y + self.height/2) * self.image_height)
        self.polygon = Polygon([(self.x_1, self.y_1), (self.x_2, self.y_1), (self.x_2, self.y_2), (self.x_1, self.y_2)])
        return self

    


class LabelTiler:
    def __init__(
        self,
        overlap,
        tile_size,
        image_size,
        num_tiles,
        label_path=None,
        save_dir=None,
        og_tile_size=None,
    ):
        self.overlap = overlap
        self.tile_size = tile_size
        self.image_size = image_size
        self.num_tiles = num_tiles
        self.label_path = label_path
        self.save_dir = save_dir
        self.og_tile_size = og_tile_size

    def tile_labels(self, label_path: Path, tiles: list[Tile]):
        """Stuff.

        Args:
            image_path: Stuff

        Returns:
            Stuff
        """
        labels = self.read_labels(label_path)

        tiles_to_labels: dict[Tile, list[Label]]
        for tile in tiles:
            # Check if labels are within
            for label in labels:
                # Check if any labels are within a tile
                if not self.label_in_tile(tile, label):
                    continue





    def label_in_tile(self, tile: Tile, label: Label) -> bool:
        return label.polygon.intersects(tile.polygon)

    def read_labels(self, label_path: Path) -> list[Label]:
        with open(label_path) as f:
            labels = []
            for line in f:
                label = line.strip().split()
                if len(label) == 5:
                    object_class, x, y, width, height = map(int, label)
                    labels.append(Label(object_class=object_class, x=x, y=y, width=width, height=height, image_width=self.image_size[0], image_height=self.image_size[1]))
        return labels

    def save_tiles(self, save_dir: Path):
        """Stuff.

        Args:
            save_dir: Stuff

        Returns:
            Stuff
        """
        pass
