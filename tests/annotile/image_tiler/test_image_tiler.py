from pathlib import Path
from annotile.image_tiler.image_tiler import ImageTiler

# Create a path to tests/data
data_path = (Path(__file__).parent / "../tests/data").resolve()
print(data_path)

tiler = ImageTiler(
    overlap=0.2, tile_size=(333, 333), image_size=(1000, 1000), num_tiles=(2, 2)
)
tiler.process_image(
    data_path / "image.jpg",
    data_path / "tiles",
    num_tiles=(3, 3),
    og_tile_size=(500, 500),
)
