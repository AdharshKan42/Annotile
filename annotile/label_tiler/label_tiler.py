from pathlib import Path


class LabelTiler:
    def __init__(self, overlap_pct, tile_size):
        self.overlap_pct = overlap_pct
        self.tile_size = tile_size

    def tile_label(self, image_path: Path):
        pass

    def save_tiles(self, save_dir: Path):
        pass
