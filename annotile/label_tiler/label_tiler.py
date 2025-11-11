from collections import defaultdict
from pathlib import Path
from typing import Self

from pydantic import BaseModel, model_validator
from shapely import Polygon

from annotile.image_tiler.image_tiler import Tile


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
    def _convert_xywh_to_bbox(self) -> Self:
        self.x_1 = int((self.x - self.width / 2) * self.image_width)
        self.y_1 = int((self.y - self.height / 2) * self.image_height)
        self.x_2 = int((self.x + self.width / 2) * self.image_width)
        self.y_2 = int((self.y + self.height / 2) * self.image_height)
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

        tiles_to_labels: defaultdict[Tile, list[Label]] = defaultdict(list)

        for tile in tiles:
            tile_labels = []
            for label in labels:
                # Check if any labels are within a tile
                if self.label_in_tile(tile, label):
                    l_clipped = label.polygon.intersection(tile.polygon)
                    # Convert to polygon coords relative to tile
                    minx, miny, maxx, maxy = l_clipped.bounds
                    label_in_tile = Label(
                        object_class=label.object_class,
                        x=((minx + maxx) / 2 - tile.x_1) / tile.tile_size[0],
                        y=((miny + maxy) / 2 - tile.y_1) / tile.tile_size[1],
                        width=(maxx - minx) / tile.tile_size[0],
                        height=(maxy - miny) / tile.tile_size[1],
                        image_width=tile.tile_size[0],
                        image_height=tile.tile_size[1],
                    )
                    tile_labels.append(label_in_tile)
            tiles_to_labels[tile] = tile_labels

        return tiles_to_labels

    def label_in_tile(self, tile: Tile, label: Label) -> bool:
        return label.polygon.intersects(tile.polygon)

    def read_labels(self, label_path: Path) -> list[Label]:
        with open(label_path) as f:
            labels = []
            for line in f:
                label = line.strip().split()
                if len(label) == 5:
                    object_class, x, y, width, height = map(int, label)
                    labels.append(
                        Label(
                            object_class=object_class,
                            x=x,
                            y=y,
                            width=width,
                            height=height,
                            image_width=self.image_size[0],
                            image_height=self.image_size[1],
                        )
                    )
        return labels

    def save_tiles(self, tiles_to_labels: dict[Tile, list[Label]], save_dir: Path):
        """Save tile labels to disk.

        Args:
            save_dir: Stuff

        Returns:
            Stuff
        """
        save_dir.mkdir(parents=True, exist_ok=True)
        for tile, labels in tiles_to_labels.items():
            tile_label_path = save_dir / f"{tile.image_path.stem}_{tile.position_in_image}.txt"
            with open(tile_label_path, "w") as f:
                for label in labels:
                    f.write(f"{label.object_class} {label.x} {label.y} {label.width} {label.height}\n")
