# ML Engineer Test - Blueprint Walls and Rooms

This repository provides a baseline implementation for:
- Wall detection in architectural blueprint images
- Room detection from detected wall structures
- API serving for inference using FastAPI

## Tech Stack

- PyTorch / Ultralytics YOLOv8 (expected for training)
- ONNX Runtime (expected for deployment inference)
- OpenCV, Pillow, NumPy for image processing
- FastAPI + Pydantic for serving and schema validation
- Docker for reproducible runtime

## Current Implementation

The API is fully runnable and includes:
- `type=wall`: returns detected wall contours and annotated image
- `type=room`: returns room polygons from wall topology + annotated image
- `type=tables`: returns detected table/grid regions and annotated image
- `type=page_info`: returns page dimensions with wall context

If `models/wall_detector.onnx` exists, that path is ready for integration.  
If not, the project uses a CV heuristic baseline (threshold + morphology + contours), so you can run and demonstrate end-to-end immediately.

## Project Structure

```text
app/
  main.py         # FastAPI server and endpoint handlers
  inference.py    # wall detector (model hook + CV fallback)
  postprocess.py  # room extraction from wall mask
  schemas.py      # request/response schemas
  utils.py        # drawing and serialization helpers
scripts/
  train_wall_yolo.py
  export_wall_onnx.py
  test_inference.py
  validate_dataset.py
  batch_pdf_inference.py
data/
  blueprints.yaml
  images/{train,val,test}
  labels/{train,val,test}
Dockerfile
requirements.txt
```

## Local Run

### 1) Create and activate virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows PowerShell
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Run API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 3000 --reload
```

### 4) Health check

```bash
curl http://localhost:3000/health
```

Response includes the active wall backend:
- `cv-heuristic` if no model is available
- `ultralytics-pytorch` if `models/wall_detector.pt` exists
- `ultralytics-onnx` if `models/wall_detector.onnx` exists

## Docker Run

```bash
docker build -t blueprint-cv-api .
docker run --rm -p 3000:3000 blueprint-cv-api
```

## Example cURL

```bash
curl -X POST -F "image=@extracted_page_xyz.png" "http://localhost:3000/run-inference?type=wall"
curl -X POST -F "image=@extracted_page_xyz.png" "http://localhost:3000/run-inference?type=room"
curl -X POST -F "image=@extracted_page_xyz.png" "http://localhost:3000/run-inference?type=page_info"
curl -X POST -F "image=@extracted_page_xyz.png" "http://localhost:3000/run-inference?type=tables"
```

The API also accepts PDF uploads (`application/pdf`) and automatically runs inference on page 1 rendered at 300 DPI.

Example (PDF):

```bash
curl -X POST -F "image=@plan.pdf;type=application/pdf" "http://localhost:3000/run-inference?type=wall"
```

## API Response

`/run-inference` returns JSON with:
- `type`
- `annotated_image_base64` (PNG encoded as Base64)
- wall and room metadata (`walls`, `rooms`)
- optional message for placeholders

## Training and ONNX Integration Notes

Recommended production flow:
1. Annotate wall masks/boxes in blueprint dataset.
2. Train YOLOv8 (`ultralytics`) for wall class.
3. Export to ONNX.
4. Save artifact to `models/wall_detector.onnx` (or `.pt`).

### Training example

```bash
python scripts/validate_dataset.py --data-root data
python scripts/train_wall_yolo.py --data data/blueprints.yaml --model yolov8n-seg.pt --epochs 50 --imgsz 1024
```

Expected YOLO folder layout:

```text
data/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
```

Each image in `images/<split>` should have a label file with the same stem in `labels/<split>`.

### Export example

```bash
python scripts/export_wall_onnx.py --weights runs/wall_train/exp/weights/best.pt --imgsz 1024 --target models/wall_detector.onnx
```

### Inference test helper

```bash
python scripts/test_inference.py --image extracted_page_xyz.png --type room --out outputs/room_result.png
```

You can also pass a PDF file:

```bash
python scripts/test_inference.py --image plan.pdf --type wall --out outputs/wall_from_pdf.png
```

Batch run all PDFs in a folder:

```bash
python scripts/batch_pdf_inference.py --pdf-dir datasets/pdfs --type room --out-dir outputs/batch_pdf_room
```

This command saves, for each PDF:
- `<name>.png` annotated output image
- `<name>.json` response metadata (without base64 image)
- `_summary.json` consolidated run summary

## Submission Checklist

- [ ] Add your trained model artifacts
- [ ] Include sample input/output results
- [ ] Verify `type=wall` and `type=room` end-to-end
- [ ] Keep Docker build and run commands working
- [ ] Share repository access with:
  - `vhaine-tb`
  - `gabrielreis-tb`
  - `dhcsouza-tb`
