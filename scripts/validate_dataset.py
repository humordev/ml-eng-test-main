from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate YOLO dataset image/label consistency.")
    parser.add_argument("--data-root", default="data", help="Dataset root directory.")
    return parser.parse_args()


def iter_images(image_dir: Path) -> Iterable[Path]:
    for path in image_dir.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def validate_split(data_root: Path, split: str) -> tuple[int, int]:
    image_dir = data_root / "images" / split
    label_dir = data_root / "labels" / split

    if not image_dir.exists():
        raise FileNotFoundError(f"Missing directory: {image_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"Missing directory: {label_dir}")

    images = sorted(iter_images(image_dir))
    missing_labels = 0
    checked = 0

    for image_path in images:
        label_path = label_dir / f"{image_path.stem}.txt"
        checked += 1
        if not label_path.exists():
            print(f"[WARN] Missing label for image: {image_path.name}")
            missing_labels += 1

    return checked, missing_labels


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)

    total_images = 0
    total_missing = 0

    for split in SPLITS:
        checked, missing = validate_split(data_root, split)
        total_images += checked
        total_missing += missing
        print(f"[{split}] images={checked} missing_labels={missing}")

    print(f"[TOTAL] images={total_images} missing_labels={total_missing}")
    if total_missing > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
