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
    unmatched_annotations: List[Path] = []

    @model_validator(mode="after")
    def process_files(self):
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
