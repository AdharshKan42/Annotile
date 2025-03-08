# Load Images and Annotations into memory
from pathlib import Path
from typing import List, Tuple
from pydantic import BaseModel, model_validator


# Read yolo images
def get_yolo_image_paths(
    path: Path, ext: List[str] = [".jpg", ".png", ".jpeg"]
) -> List[Path]:
    if not path.is_dir():
        raise NotADirectoryError(f"{path} is not a directory.")

    return [
        file for file in path.iterdir() if file.suffix.lower() in ext and file.is_file()
    ]


# Read yolo annotations
def get_yolo_annotation_paths(path: Path, ext: List[str] = [".txt"]) -> List[Path]:
    if not path.is_dir():
        raise NotADirectoryError(f"{path} is not a directory.")

    return [
        file for file in path.iterdir() if file.suffix.lower() in ext and file.is_file()
    ]


# Store image and annotation paths
class Dataloader(BaseModel):
    image_paths: List[Path]
    annotation_paths: List[Path]

    # These will be automatically filled after validation
    paired: List[Tuple[Path, Path]] = []
    unmatched_images: List[Path] = []

    @model_validator(mode="after")
    def process_files(self):
        annotation_map = {ann.stem: ann for ann in self.annotation_paths}
        paired = []
        unmatched = []

        for img in self.image_paths:
            annotation = annotation_map.get(
                img.stem
            )  # Match by filename (without extension)
            if annotation:
                paired.append((img, annotation))
            else:
                unmatched.append(img)

        # Assign the processed values to the instance
        self.paired = paired
        self.unmatched_images = unmatched
        return self
