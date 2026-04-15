from __future__ import annotations

import argparse
from pathlib import Path
from shutil import copy2

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export trained YOLO model to ONNX.")
    parser.add_argument("--weights", required=True, help="Path to trained .pt weights.")
    parser.add_argument("--imgsz", type=int, default=1024, help="Export image size.")
    parser.add_argument(
        "--target",
        default="models/wall_detector.onnx",
        help="Target ONNX output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    model = YOLO(str(weights))
    onnx_path = Path(model.export(format="onnx", imgsz=args.imgsz))

    target = Path(args.target)
    target.parent.mkdir(parents=True, exist_ok=True)
    copy2(onnx_path, target)
    print(f"Exported ONNX model to: {target}")


if __name__ == "__main__":
    main()
