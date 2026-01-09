# Load Images and Annotations into memory
from pathlib import Path

from annotile.dataloader.dto import DataloaderResult


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


def load_dataset(image_dir: Path, annotation_dir: Path | None = None) -> DataloaderResult:
    """Load and pair images with annotations.

    Args:
        image_dir: Directory containing images.
        annotation_dir: Optional directory containing annotations.

    Returns:
        DataloaderResult containing paired and unmatched files.
    """
    image_paths = get_yolo_image_paths(image_dir)
    
    if annotation_dir is None:
        # Image-only mode
        return DataloaderResult(
            paired=[],
            unmatched_images=image_paths,
            unmatched_annotations=[],
        )
    
    annotation_paths = get_yolo_annotation_paths(annotation_dir)
    
    # Match images with annotations
    annotation_map = {ann.stem: ann for ann in annotation_paths}
    image_map = {img.stem: img for img in image_paths}
    
    annotation_set = set(annotation_map.keys())
    image_set = set(image_map.keys())
    
    # Find paired and unmatched
    paired_stems = image_set.intersection(annotation_set)
    paired = [(image_map[stem], annotation_map[stem]) for stem in paired_stems]
    
    unmatched_image_stems = image_set - annotation_set
    unmatched_images = [image_map[stem] for stem in unmatched_image_stems]
    
    unmatched_annotation_stems = annotation_set - image_set
    unmatched_annotations = [annotation_map[stem] for stem in unmatched_annotation_stems]
    
    return DataloaderResult(
        paired=paired,
        unmatched_images=unmatched_images,
        unmatched_annotations=unmatched_annotations,
    )
