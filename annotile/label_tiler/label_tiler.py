from pathlib import Path

from pydantic import BaseModel

from annotile.image_tiler.image_tiler import Tile


class Label(BaseModel):
    object_class: int
    x: int
    y: int
    width: int
    height: int


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

    def read_labels(self, label_path: Path) -> list[Label]:
        with open(label_path) as f:
            labels = []
            for line in f:
                label = line.strip().split()
                if len(label) == 5:
                    object_class, x, y, width, height = map(int, label)
                    labels.append(Label(object_class=object_class, x=x, y=y, width=width, height=height))
        return labels

    def save_tiles(self, save_dir: Path):
        """Stuff.

        Args:
            save_dir: Stuff

        Returns:
            Stuff
        """
        pass
