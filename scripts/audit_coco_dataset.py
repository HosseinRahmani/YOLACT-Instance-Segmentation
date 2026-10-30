"""Audit converted COCO polygon annotations and save visual samples.

Expected layout::

    data/images/{train,val,test}/...
    data/annotations/instances_{train,val,test}.json

The audit checks image references, IDs, categories, boxes, polygons, and exact
duplicate image files across splits. It writes an audit_summary.json and a few
overlay images for each split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_split(
    split: str,
    data_root: Path,
    annotation_dir: Path,
    output_dir: Path,
    samples_per_split: int,
    hashes_to_locations: dict[str, list[dict[str, str]]],
) -> dict[str, Any]:
    annotation_path = annotation_dir / f"instances_{split}.json"
    image_root = data_root / "images" / split
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Annotation JSON not found: {annotation_path}")
    if not image_root.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_root}")

    with annotation_path.open("r", encoding="utf-8") as file:
        coco = json.load(file)

    categories = {int(item["id"]): item["name"] for item in coco.get("categories", [])}
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    image_ids = [int(item["id"]) for item in images]
    annotation_ids = [int(item["id"]) for item in annotations]
    issues: list[str] = []
    if len(image_ids) != len(set(image_ids)):
        issues.append("duplicate image IDs")
    if len(annotation_ids) != len(set(annotation_ids)):
        issues.append("duplicate annotation IDs")

    image_by_id = {int(item["id"]): item for item in images}
    image_paths: dict[int, Path] = {}
    missing_images = []
    exact_duplicates = []
    for record in images:
        image_id = int(record["id"])
        image_path = image_root / record["file_name"]
        image_paths[image_id] = image_path
        if not image_path.is_file():
            missing_images.append(record["file_name"])
            continue
        digest = sha256(image_path)
        previous_locations = hashes_to_locations.setdefault(digest, [])
        if previous_locations:
            exact_duplicates.append(
                {"file": record["file_name"], "also_seen_as": previous_locations.copy()}
            )
        previous_locations.append({"split": split, "file": record["file_name"]})

    annotations_per_category: Counter[str] = Counter()
    invalid_annotations = []
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in annotations:
        annotation_id = annotation.get("id", "?")
        image_id = int(annotation.get("image_id", -1))
        category_id = int(annotation.get("category_id", -1))
        bbox = annotation.get("bbox", [])
        segmentation = annotation.get("segmentation", [])
        reasons = []
        image_record = image_by_id.get(image_id)
        if image_record is None:
            reasons.append("unknown image_id")
        if category_id not in categories:
            reasons.append("unknown category_id")
        if len(bbox) != 4:
            reasons.append("bbox must contain x, y, width, height")
        elif image_record is not None:
            x, y, width, height = [float(value) for value in bbox]
            image_width, image_height = int(image_record["width"]), int(image_record["height"])
            if width <= 0 or height <= 0:
                reasons.append("bbox has non-positive width or height")
            if x < 0 or y < 0 or x + width > image_width + 1e-3 or y + height > image_height + 1e-3:
                reasons.append("bbox is outside image bounds")
        if not isinstance(segmentation, list) or not segmentation:
            reasons.append("segmentation polygon is missing")
        else:
            for polygon in segmentation:
                if not isinstance(polygon, list) or len(polygon) < 6 or len(polygon) % 2:
                    reasons.append("polygon must contain at least 3 x/y points")
                    break
        if float(annotation.get("area", 0)) <= 0:
            reasons.append("area must be positive")

        if reasons:
            invalid_annotations.append({"annotation_id": annotation_id, "reasons": reasons})
        else:
            annotations_per_category[categories[category_id]] += 1
            annotations_by_image[image_id].append(annotation)

    if missing_images:
        issues.append(f"{len(missing_images)} image files referenced by JSON are missing")
    if invalid_annotations:
        issues.append(f"{len(invalid_annotations)} invalid annotations")
    if exact_duplicates:
        issues.append(f"{len(exact_duplicates)} exact duplicate image(s) found across splits")

    sample_dir = output_dir / split
    sample_dir.mkdir(parents=True, exist_ok=True)
    saved_samples = 0
    for record in images:
        if saved_samples >= samples_per_split:
            break
        image_id = int(record["id"])
        image_path = image_paths[image_id]
        if not image_path.is_file():
            continue
        with Image.open(image_path) as source:
            image = source.convert("RGBA")
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        for annotation in annotations_by_image.get(image_id, []):
            category_name = categories[int(annotation["category_id"])]
            for polygon in annotation["segmentation"]:
                points = list(zip(polygon[0::2], polygon[1::2]))
                draw.polygon(points, fill=(0, 180, 255, 55), outline=(0, 110, 220, 255), width=3)
            x, y, width, height = annotation["bbox"]
            draw.rectangle((x, y, x + width, y + height), outline=(255, 55, 55, 255), width=3)
            draw.text((x, max(0, y - 16)), category_name, fill=(255, 55, 55, 255), stroke_width=2, stroke_fill=(255, 255, 255, 255))
        rendered = Image.alpha_composite(image, overlay).convert("RGB")
        rendered.save(sample_dir / f"{Path(record['file_name']).stem}_overlay.jpg", quality=92)
        saved_samples += 1

    total_instances = len(annotations)
    return {
        "split": split,
        "images": len(images),
        "instances": total_instances,
        "instances_per_class": dict(annotations_per_category),
        "missing_images": missing_images,
        "invalid_annotations": invalid_annotations,
        "exact_duplicates": exact_duplicates,
        "issues": issues,
        "visual_samples_saved": saved_samples,
        "visual_samples_directory": str(sample_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--annotation-dir", type=Path, default=Path("data/annotations"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/data_audit"))
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--samples-per-split", type=int, default=6)
    args = parser.parse_args()

    hashes_to_locations: dict[str, list[dict[str, str]]] = {}
    summaries = [
        audit_split(
            split,
            args.data_root,
            args.annotation_dir,
            args.output_dir,
            args.samples_per_split,
            hashes_to_locations,
        )
        for split in args.splits
    ]
    report_path = args.output_dir / "audit_summary.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(summaries, file, indent=2)

    for summary in summaries:
        print(
            f"{summary['split']}: {summary['images']} images, {summary['instances']} instances, "
            f"{len(summary['invalid_annotations'])} invalid annotations, "
            f"{len(summary['missing_images'])} missing images, "
            f"{len(summary['exact_duplicates'])} exact duplicates"
        )
        if summary["issues"]:
            print("  Issues: " + "; ".join(summary["issues"]))
    print(f"Audit report: {report_path.resolve()}")
    print(f"Overlay samples: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
