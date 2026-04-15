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
