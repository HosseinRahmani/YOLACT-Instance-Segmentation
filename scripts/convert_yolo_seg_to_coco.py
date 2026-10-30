"""Convert Ultralytics YOLO segmentation labels to COCO instance annotations.

Expected source layout (relative to --data-root)::

    images/{train,val,test}/<image files>
    labels/{train,val,test}/<matching .txt files>

Each label line is ``class_id x1 y1 x2 y2 ...`` with normalized polygon
coordinates, as used by Ultralytics segmentation datasets. Images without a
label file are retained as background images.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image
import yaml


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def load_class_names(dataset_yaml: Path) -> list[str]:
    with dataset_yaml.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    names: Any = config.get("names")
    if isinstance(names, dict):
        names = [names[key] for key in sorted(names, key=lambda key: int(key))]
    if not isinstance(names, list) or not names:
        raise ValueError(f"Expected a non-empty 'names' list or mapping in {dataset_yaml}")
    return [str(name) for name in names]


def polygon_area(points: list[float]) -> float:
    vertices = list(zip(points[0::2], points[1::2]))
    return abs(
        sum(
            x1 * y2 - x2 * y1
            for (x1, y1), (x2, y2) in zip(vertices, vertices[1:] + vertices[:1])
        )
    ) / 2.0


def read_label_file(label_path: Path, width: int, height: int, names: list[str]):
    """Return parsed (class index, COCO polygon, bbox, area) rows."""
    parsed = []
    with label_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            fields = line.strip().split()
            if not fields:
                continue
            if len(fields) < 7 or len(fields[1:]) % 2:
                raise ValueError(
                    f"{label_path}:{line_number}: expected class ID and at least 3 (x, y) points"
                )

            try:
                class_index = int(fields[0])
                normalized = [float(value) for value in fields[1:]]
            except ValueError as error:
                raise ValueError(f"{label_path}:{line_number}: non-numeric label value") from error

            if class_index < 0 or class_index >= len(names):
                raise ValueError(
                    f"{label_path}:{line_number}: class ID {class_index} has no entry in names"
                )
            if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in normalized):
                raise ValueError(
                    f"{label_path}:{line_number}: normalized coordinates must be finite and in [0, 1]"
                )

            points = []
            for x, y in zip(normalized[0::2], normalized[1::2]):
                points.extend((x * width, y * height))

            # YOLO files may repeat the first point as the final point. COCO
            # polygon segmentation does not need that duplicate closing point.
            if len(points) >= 6 and points[:2] == points[-2:]:
                points = points[:-2]
            if len(points) < 6:
                raise ValueError(f"{label_path}:{line_number}: polygon has fewer than 3 points")

            xs, ys = points[0::2], points[1::2]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            area = polygon_area(points)
            if area <= 0 or x_max <= x_min or y_max <= y_min:
                raise ValueError(f"{label_path}:{line_number}: polygon has zero area")

            parsed.append((class_index, points, [x_min, y_min, x_max - x_min, y_max - y_min], area))
    return parsed


def convert_split(data_root: Path, split: str, names: list[str], output_dir: Path) -> dict[str, int]:
    image_dir = data_root / "images" / split
    label_dir = data_root / "labels" / split
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not label_dir.is_dir():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")

    image_paths = sorted(
        path for path in image_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_paths:
        raise FileNotFoundError(f"No supported images found in {image_dir}")

    coco = {
        "info": {"description": "Package-Seg converted from YOLO segmentation format"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {"id": index + 1, "name": name, "supercategory": "package"}
            for index, name in enumerate(names)
        ],
    }

    annotation_id = 1
    missing_label_files = 0
    for image_id, image_path in enumerate(image_paths, start=1):
        with Image.open(image_path) as image:
            width, height = image.size
        relative_image = image_path.relative_to(image_dir).as_posix()
        coco["images"].append(
            {
                "id": image_id,
                "file_name": relative_image,
                "width": width,
                "height": height,
            }
        )

        relative_label = image_path.relative_to(image_dir).with_suffix(".txt")
        label_path = label_dir / relative_label
        if not label_path.is_file():
            missing_label_files += 1
            continue

        for class_index, polygon, bbox, area in read_label_file(label_path, width, height, names):
            coco["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_index + 1,
                    "segmentation": [polygon],
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"instances_{split}.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(coco, file, indent=2)

    return {
        "images": len(coco["images"]),
        "instances": len(coco["annotations"]),
        "images_without_label_file": missing_label_files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--dataset-yaml", type=Path, default=Path("data/package-seg.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/annotations"))
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = parser.parse_args()

    names = load_class_names(args.dataset_yaml)
    for split in args.splits:
        summary = convert_split(args.data_root, split, names, args.output_dir)
        print(
            f"{split}: {summary['images']} images, {summary['instances']} instances, "
            f"{summary['images_without_label_file']} images without label files"
        )
    print(f"COCO annotations written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
