from __future__ import annotations

import argparse
import base64
import json
import mimetypes
from pathlib import Path
from typing import Any

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference for all PDF files in a folder and save results."
    )
    parser.add_argument("--pdf-dir", required=True, help="Directory containing PDF files.")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan subfolders for PDF files.",
    )
    parser.add_argument(
        "--type",
        default="wall",
        choices=["wall", "room", "page_info", "tables"],
        help="Inference type.",
    )
    parser.add_argument("--url", default="http://localhost:3000/run-inference", help="Inference URL.")
    parser.add_argument(
        "--out-dir",
        default="outputs/batch_pdf",
        help="Output folder for annotated images and JSON metadata.",
    )
    return parser.parse_args()


def run_single_pdf(pdf_path: Path, infer_type: str, url: str) -> dict[str, Any]:
    with pdf_path.open("rb") as f:
        content_type = mimetypes.guess_type(str(pdf_path))[0] or "application/pdf"
        response = requests.post(
            url,
            params={"type": infer_type},
            files={"image": (pdf_path.name, f, content_type)},
            timeout=180,
        )
    response.raise_for_status()
    return response.json()


def save_result(payload: dict[str, Any], pdf_path: Path, out_dir: Path) -> None:
    stem = pdf_path.stem
    encoded_image = payload.get("annotated_image_base64")
    if not encoded_image:
        raise ValueError(f"Missing 'annotated_image_base64' for {pdf_path.name}")

    image_bytes = base64.b64decode(encoded_image)
    (out_dir / f"{stem}.png").write_bytes(image_bytes)

    payload_to_save = dict(payload)
    payload_to_save.pop("annotated_image_base64", None)
    (out_dir / f"{stem}.json").write_text(json.dumps(payload_to_save, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    pdf_dir = Path(args.pdf_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pdf_dir.exists() or not pdf_dir.is_dir():
        raise FileNotFoundError(f"PDF directory does not exist: {pdf_dir}")

    if args.recursive:
        pdf_files = sorted([p for p in pdf_dir.rglob("*.pdf") if p.is_file()])
    else:
        pdf_files = sorted([p for p in pdf_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"])
    if not pdf_files:
        recursive_hint = " (recursive enabled)" if args.recursive else ""
        raise FileNotFoundError(f"No PDF files found in: {pdf_dir}{recursive_hint}")

    processed = 0
    failed = 0
    summary: list[dict[str, Any]] = []

    for pdf_path in pdf_files:
        try:
            payload = run_single_pdf(pdf_path, infer_type=args.type, url=args.url)
            save_result(payload, pdf_path=pdf_path, out_dir=out_dir)
            processed += 1
            summary.append(
                {
                    "file": pdf_path.name,
                    "status": "ok",
                    "wall_count": payload.get("wall_count", 0),
                    "room_count": payload.get("room_count", 0),
                    "message": payload.get("message"),
                }
            )
            print(f"[OK] {pdf_path.name}")
        except Exception as exc:
            failed += 1
            summary.append({"file": pdf_path.name, "status": "failed", "error": str(exc)})
            print(f"[FAIL] {pdf_path.name}: {exc}")

    summary_path = out_dir / "_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "pdf_total": len(pdf_files),
                "processed": processed,
                "failed": failed,
                "summary_file": str(summary_path),
            },
            indent=2,
        )
    )

    if failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
