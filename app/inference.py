from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class WallPrediction:
    confidence: float
    contour: np.ndarray


class WallDetector:
    """
    Baseline detector:
    - If model artifacts are present, this class is the extension point for ONNX/YOLO inference.
    - Without artifacts, it uses a CV heuristic to extract thick line structures as walls.
    """

    def __init__(self, model_dir: str = "models") -> None:
        self.model_dir = Path(model_dir)
        self.onnx_path = self.model_dir / "wall_detector.onnx"
        self.pt_path = self.model_dir / "wall_detector.pt"
        self.model_source = self._resolve_model_source()
        self.yolo_model = YOLO(str(self.model_source)) if self.model_source is not None else None

    def predict_walls(self, bgr_image: np.ndarray) -> List[WallPrediction]:
        if self.yolo_model is not None:
            return self._predict_with_yolo_model(bgr_image)
        return self._predict_with_cv_heuristic(bgr_image)

    def _resolve_model_source(self) -> Path | None:
        if self.onnx_path.exists():
            return self.onnx_path
        if self.pt_path.exists():
            return self.pt_path
        return None

    def get_backend(self) -> str:
        if self.model_source is None:
            return "cv-heuristic"
        suffix = self.model_source.suffix.lower()
        if suffix == ".onnx":
            return "ultralytics-onnx"
        if suffix == ".pt":
            return "ultralytics-pytorch"
        return "ultralytics-unknown"

    def _predict_with_yolo_model(self, bgr_image: np.ndarray) -> List[WallPrediction]:
        if self.yolo_model is None:
            return []

        results = self.yolo_model.predict(source=bgr_image, verbose=False, conf=0.2, iou=0.5)
        if not results:
            return []

        result = results[0]
        predictions: List[WallPrediction] = []

        # Prefer segmentation masks for wall boundaries.
        if result.masks is not None and result.masks.xy is not None:
            for idx, polygon in enumerate(result.masks.xy):
                if polygon.shape[0] < 3:
                    continue
                contour = polygon.astype(np.int32).reshape(-1, 1, 2)
                confidence = (
                    float(result.boxes.conf[idx].item())
                    if result.boxes is not None and idx < len(result.boxes.conf)
                    else 0.8
                )
                predictions.append(WallPrediction(confidence=confidence, contour=contour))
            return predictions

        # Fallback to boxes if segmentation heads are not available.
        if result.boxes is not None:
            for i in range(len(result.boxes)):
                conf = float(result.boxes.conf[i].item())
                x1, y1, x2, y2 = result.boxes.xyxy[i].int().tolist()
                contour = np.array(
                    [[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]],
                    dtype=np.int32,
                )
                predictions.append(WallPrediction(confidence=conf, contour=contour))

        return predictions

    def _predict_with_cv_heuristic(self, bgr_image: np.ndarray) -> List[WallPrediction]:
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        binary_inv = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            8,
        )

        # Emphasize blueprint linear structures
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(binary_inv, cv2.MORPH_CLOSE, kernel, iterations=2)
        dilated = cv2.dilate(closed, kernel, iterations=1)

        # Strengthen horizontal/vertical wall segments.
        h, w = gray.shape
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, w // 80), 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, h // 80)))
        horizontal = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, h_kernel, iterations=1)
        vertical = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, v_kernel, iterations=1)
        wall_like = cv2.bitwise_or(dilated, cv2.bitwise_or(horizontal, vertical))

        contours, _ = cv2.findContours(wall_like, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_area = max(100, int((h * w) * 0.0002))
        predictions: List[WallPrediction] = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            # Keep long-ish, wall-like components.
            x, y, bw, bh = cv2.boundingRect(contour)
            aspect_ratio = max(bw / max(1, bh), bh / max(1, bw))
            fill_ratio = area / max(1.0, float(bw * bh))
            if aspect_ratio < 1.15 and fill_ratio > 0.75:
                continue

            approx = cv2.approxPolyDP(contour, epsilon=2.0, closed=True)
            confidence = float(min(0.99, 0.5 + (area / (h * w)) * 10))
            predictions.append(WallPrediction(confidence=confidence, contour=approx))

        return predictions


def build_wall_mask(image_shape: Tuple[int, int], walls: List[WallPrediction], thickness: int = 3) -> np.ndarray:
    mask = np.zeros(image_shape, dtype=np.uint8)
    for wall in walls:
        cv2.drawContours(mask, [wall.contour], contourIdx=-1, color=255, thickness=thickness)
    return mask
