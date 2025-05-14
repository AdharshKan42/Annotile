from annotile.dataloader.dataloader import (
    Dataloader,
    get_yolo_annotation_paths,
    get_yolo_image_paths,
)
from pathlib import Path

# Create a path to tests/data
data_path = (Path(__file__).parent / "../tests/data").resolve()
print(data_path)

image_paths = get_yolo_image_paths(data_path)
annotation_paths = get_yolo_annotation_paths(data_path)

dataloader = Dataloader(image_paths=image_paths, annotation_paths=annotation_paths)
print(dataloader.paired)
print("**" * 20)
print(dataloader.unmatched_images)
print("**" * 20)
print(dataloader.unmatched_annotations)
print("**" * 20)
print(dataloader.model_dump())
