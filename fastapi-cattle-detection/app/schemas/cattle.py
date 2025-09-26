from pydantic import BaseModel
from typing import Optional

class CattleUploadRequest(BaseModel):
    image: str  # Path to the uploaded image

class CattlePredictionResponse(BaseModel):
    breed: str
    height_cm: Optional[float]
    width_cm: Optional[float]
    weight_kg: Optional[float]
    milk_yield_l_per_day: Optional[float]