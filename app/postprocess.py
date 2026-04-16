from __future__ import annotations

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np


@dataclass
class RoomPrediction:
    room_id: int
    area_px: int
    contour: np.ndarray


@dataclass
class TablePrediction:
    confidence: float
    contour: np.ndarray


def detect_rooms_from_walls(wall_mask: np.ndarray, min_area: int = 500) -> List[RoomPrediction]:
    """
    Basic room extraction:
    - Treat walls as separators.
    - Connected components on free space become room candidates.
    """
    # Close tiny gaps to avoid leaking between rooms.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    wall_closed = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    free_space = cv2.bitwise_not(wall_closed)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(free_space, connectivity=8)

    rooms: List[RoomPrediction] = []
    room_id = 1

    for idx in range(1, num_labels):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        component_mask = np.where(labels == idx, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon=max(1.0, 0.01 * perimeter), closed=True)
        rooms.append(RoomPrediction(room_id=room_id, area_px=area, contour=approx))
        room_id += 1

    return rooms


def detect_tables_from_image(bgr_image: np.ndarray) -> List[TablePrediction]:
    """
    Heuristic table detector for blueprint/schedule pages.
    Detects large grid-like structures built by horizontal/vertical lines.
    """
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    bin_inv = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        9,
    )

    h, w = gray.shape
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, w // 30), 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, h // 30)))

    horizontal = cv2.morphologyEx(bin_inv, cv2.MORPH_OPEN, h_kernel, iterations=1)
    vertical = cv2.morphologyEx(bin_inv, cv2.MORPH_OPEN, v_kernel, iterations=1)
    grid = cv2.bitwise_or(horizontal, vertical)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    grid = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = max(2000, int((h * w) * 0.002))
    tables: List[TablePrediction] = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        x, y, bw, bh = cv2.boundingRect(contour)
        if bw < max(120, w // 10) or bh < max(80, h // 15):
            continue

        roi = grid[y : y + bh, x : x + bw]
        line_density = float(cv2.countNonZero(roi)) / max(1.0, float(bw * bh))
        if line_density < 0.03:
            continue

        approx = cv2.approxPolyDP(contour, epsilon=3.0, closed=True)
        confidence = float(min(0.99, 0.5 + line_density * 2.0))
        tables.append(TablePrediction(confidence=confidence, contour=approx))

    return tables
