from typing import List, Literal, Optional

from pydantic import BaseModel, Field


InferenceType = Literal["wall", "room", "page_info", "tables"]


class Point(BaseModel):
    x: int
    y: int


class Polygon(BaseModel):
    points: List[Point]


class WallDetection(BaseModel):
    confidence: float = Field(ge=0.0, le=1.0)
    polygon: Polygon


class RoomDetection(BaseModel):
    room_id: int
    area_px: int
    polygon: Polygon


class TableDetection(BaseModel):
    confidence: float = Field(ge=0.0, le=1.0)
    polygon: Polygon


class InferenceResponse(BaseModel):
    type: InferenceType
    annotated_image_base64: str
    wall_count: int = 0
    room_count: int = 0
    table_count: int = 0
    page_width: Optional[int] = None
    page_height: Optional[int] = None
    walls: Optional[List[WallDetection]] = None
    rooms: Optional[List[RoomDetection]] = None
    tables: Optional[List[TableDetection]] = None
    message: Optional[str] = None
