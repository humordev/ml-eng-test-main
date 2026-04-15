from __future__ import annotations

import argparse
import base64
import mimetypes
from pathlib import Path

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test API inference and save annotated output.")
    parser.add_argument("--image", required=True, help="Input file path (image or PDF).")
    parser.add_argument("--type", default="wall", choices=["wall", "room", "page_info", "tables"])
    parser.add_argument("--url", default="http://localhost:3000/run-inference", help="Inference URL.")
    parser.add_argument("--out", default="outputs/annotated.png", help="Output annotated PNG path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with image_path.open("rb") as f:
        content_type = mimetypes.guess_type(str(image_path))[0] or "application/octet-stream"
        response = requests.post(
            args.url,
            params={"type": args.type},
            files={"image": (image_path.name, f, content_type)},
            timeout=90,
        )

    response.raise_for_status()
    payload = response.json()

    encoded = payload.get("annotated_image_base64")
    if not encoded:
        raise ValueError("Response does not include 'annotated_image_base64'.")

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(base64.b64decode(encoded))
    print(f"Saved annotated image to: {output_path}")
    print(
        {
            "type": payload.get("type"),
            "wall_count": payload.get("wall_count"),
            "room_count": payload.get("room_count"),
            "message": payload.get("message"),
        }
    )


if __name__ == "__main__":
    main()
