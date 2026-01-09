from collections import defaultdict
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator
from shapely import Polygon

from annotile.tiling.image_tiler.image_tiler import Tile
from annotile.tiling.label_tiler.dto import LabelTilerConfig


class Label(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

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
    # MIGHT lose validation on Shapley Polygon and can't seralize, TODO fix
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
    """Tiles labels using configuration DTO.
    
    Receives validated parameters from ImageTiler.
    """

    def __init__(self, config: LabelTilerConfig):
        """Initialize LabelTiler with configuration DTO.

        Args:
            config: LabelTilerConfig containing all required parameters.
        """
        self.config = config

    def tile_labels(self, tiles: list[Tile]) -> dict[Tile, list[Label]]:
        """Tile labels based on image tiles.

        Args:
            tiles: List of Tile objects from ImageTiler.

        Returns:
            Dictionary mapping each tile to its labels.
        """
        labels = self.read_labels()

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
                        x=((minx + maxx) / 2 - tile.position_in_image[0]) / tile.tile_size[0],
                        y=((miny + maxy) / 2 - tile.position_in_image[1]) / tile.tile_size[1],
                        width=(maxx - minx) / tile.tile_size[0],
                        height=(maxy - miny) / tile.tile_size[1],
                        image_width=tile.tile_size[0],
                        image_height=tile.tile_size[1],
                    )
                    tile_labels.append(label_in_tile)
            tiles_to_labels[tile] = tile_labels

        return dict(tiles_to_labels)

    def label_in_tile(self, tile: Tile, label: Label) -> bool:
        return label.polygon.intersects(tile.polygon)

    def read_labels(self) -> list[Label]:
        """Read labels from the configured label path.
        
        Returns:
            List of Label objects.
        """
        with open(self.config.label_path) as f:
            labels = []
            for line in f:
                label = line.strip().split()
                if len(label) == 5:
                    object_class, x, y, width, height = map(float, label)
                    labels.append(
                        Label(
                            object_class=int(object_class),
                            x=x,
                            y=y,
                            width=width,
                            height=height,
                            image_width=self.config.image_size[0],
                            image_height=self.config.image_size[1],
                        )
                    )
        return labels

    def save_tiles(self, tiles_to_labels: dict[Tile, list[Label]]) -> None:
        """Save tile labels to disk.

        Args:
            tiles_to_labels: Dictionary mapping tiles to their labels.
        """
        save_dir = self.config.save_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for tile, labels in tiles_to_labels.items():
            tile_label_path = save_dir / f"{tile.image_path.stem}_{tile.position_in_image}.txt"
            with open(tile_label_path, "w") as f:
                for label in labels:
                    f.write(f"{label.object_class} {label.x} {label.y} {label.width} {label.height}\n")

    def process(self, tiles: list[Tile]) -> dict[Tile, list[Label]]:
        """Process labels: tile them and save.
        
        Args:
            tiles: List of Tile objects from ImageTiler.
            
        Returns:
            Dictionary mapping tiles to labels.
        """
        tiles_to_labels = self.tile_labels(tiles)
        self.save_tiles(tiles_to_labels)
        return tiles_to_labels
