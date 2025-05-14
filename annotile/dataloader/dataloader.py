# Load Images and Annotations into memory
from pathlib import Path
from typing import Self

from pydantic import BaseModel, model_validator


def get_yolo_image_paths(path: Path, ext: list[str] | None = None) -> list[Path]:
    """Grabs Pathlib Paths for images with YOLO formatting to be tiled.

    Args:
        path (Path): Path to directory containing potential YOLO images to be tiled
        ext (list[str] | None): Valid image extensions for tiling use

    Returns:
        (list[Path]) list of Paths of valid YOLO images to be used for tiling.
    """
    if ext is None:
        ext = [".jpg", ".png", ".jpeg"]
    if not path.is_dir():
        raise NotADirectoryError(f"{path} is not a directory.")

    return [file for file in path.iterdir() if file.suffix.lower() in ext and file.is_file()]


def get_yolo_annotation_paths(path: Path, ext: list[str] | None = None) -> list[Path]:
    """Grabs Pathlib Paths for labels with YOLO formatting to be tiled.

    Args:
        path (Path): Path to directory containing potential YOLO labels to be tiled
        ext (list[str] | None): Valid label extensions for tiling use

    Returns:
        (list[Path]): list of Paths of valid YOLO labels to be used for tiling.
    """
    if ext is None:
        ext = [".txt"]
    if not path.is_dir():
        raise NotADirectoryError(f"{path} is not a directory.")

    return [file for file in path.iterdir() if file.suffix.lower() in ext and file.is_file()]


class Dataloader(BaseModel):
    image_paths: list[Path]
    annotation_paths: list[Path]

    # These will be automatically filled after validation
    paired: list[tuple[Path, Path]] = []
    unmatched_images: list[Path] = []
    unmatched_annotations: list[Path] = []
    overlap: float = 0.2
    tile_size: tuple[int, int] = (512, 512)
    image_size: tuple[int, int] = (2048, 2048)
    num_tiles: tuple[int, int] = (4, 4)
    og_tile_size: tuple[int, int] = (512, 512)

    @model_validator(mode="after")
    def process_files(self) -> Self:
        """Data validator that groups images and corresponding labels together.

        Data validator that groups images and corresponding labels together.  for future
        multi-processed tiling.

        Returns:
            (Self): Instance of Dataloader with validated data
        """
        annotation_map = {ann.stem: ann for ann in self.annotation_paths}
        image_map = {img.stem: img for img in self.image_paths}

        paired = []
        unmatched_images = []
        unmatched_annotations = []

        # Check the Set difference between the two sets
        annotation_set = set(annotation_map.keys())
        image_set = set(image_map.keys())

        # Annotations without corresponding images
        unmatched_annotation_stems = annotation_set - image_set
        if len(unmatched_annotation_stems) > 0:
            unmatched_annotations = [annotation_map[stem] for stem in unmatched_annotation_stems]

        # Images without corresponding annotations
        unmatched_image_stems = image_set - annotation_set
        if len(unmatched_image_stems) > 0:
            unmatched_images = [image_map[stem] for stem in unmatched_image_stems]

        # Match images with annotations
        paired_images = image_set.intersection(annotation_set)
        paired = [(image_map[stem], annotation_map[stem]) for stem in paired_images]

        # Assign the processed values to the instance
        self.paired = paired
        self.unmatched_images = unmatched_images
        self.unmatched_annotations = unmatched_annotations

        return self
