import logging
from pathlib import Path

from annotile.tiling import Tiler, TilerConfig


def main():
    """Main entry point for the Annotile tiling system."""
    logging.basicConfig(level=logging.INFO)
    
    # Example usage with paired mode
    config = TilerConfig(
        image_dir=Path("/Users/adharshkandula/Documents/Repos/Annotile/tests/data/test_img"),
        annotation_dir=Path("/Users/adharshkandula/Documents/Repos/Annotile/tests/data/test_labels"),
        save_image_dir=Path("/Users/adharshkandula/Documents/Repos/Annotile/tests/data/tiled_images"),
        save_label_dir=Path("/Users/adharshkandula/Documents/Repos/Annotile/tests/data/tiled_labels"),
        tile_size=(500, 500),
        overlap=0.2,
        use_paired=True,
    )
    
    tiler = Tiler(config)
    tiler.tile_dataset()


if __name__ == "__main__":
    main()
