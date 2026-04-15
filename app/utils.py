from __future__ import annotations

import base64
from typing import Iterable, List

import cv2
import numpy as np

from app.inference import WallPrediction
from app.postprocess import RoomPrediction
from app.schemas import Point, Polygon, RoomDetection, WallDetection


def contour_to_polygon(contour: np.ndarray) -> Polygon:
    points: List[Point] = [Point(x=int(pt[0][0]), y=int(pt[0][1])) for pt in contour]
    return Polygon(points=points)


def walls_to_schema(walls: Iterable[WallPrediction]) -> List[WallDetection]:
    return [
        WallDetection(confidence=float(w.confidence), polygon=contour_to_polygon(w.contour))
        for w in walls
    ]


def rooms_to_schema(rooms: Iterable[RoomPrediction]) -> List[RoomDetection]:
    return [
        RoomDetection(room_id=r.room_id, area_px=int(r.area_px), polygon=contour_to_polygon(r.contour))
        for r in rooms
    ]


def encode_image_base64(image_bgr: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".png", image_bgr)
    if not ok:
        raise ValueError("Could not encode image.")
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


def draw_walls(image_bgr: np.ndarray, walls: Iterable[WallPrediction]) -> np.ndarray:
    out = image_bgr.copy()
    for wall in walls:
        cv2.drawContours(out, [wall.contour], -1, (0, 0, 255), 2)
    return out


def draw_rooms(image_bgr: np.ndarray, rooms: Iterable[RoomPrediction]) -> np.ndarray:
    out = image_bgr.copy()
    overlay = out.copy()
    for room in rooms:
        color = (
            int((room.room_id * 53) % 255),
            int((room.room_id * 97) % 255),
            int((room.room_id * 149) % 255),
        )
        cv2.drawContours(overlay, [room.contour], -1, color, thickness=-1)

        moments = cv2.moments(room.contour)
        if moments["m00"] > 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            cv2.putText(
                overlay,
                f"R{room.room_id}",
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

    cv2.addWeighted(overlay, 0.35, out, 0.65, 0, out)
    return out
