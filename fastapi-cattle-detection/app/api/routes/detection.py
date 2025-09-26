from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from app.services.detection_service import detect_objects
from app.services.breed_service import predict_breed
from app.services.weight_service import calculate_weight
from app.services.milk_service import calculate_milk_yield
from app.schemas.cattle import CattlePredictionResponse
import os

router = APIRouter()

@router.post("/upload/", response_model=CattlePredictionResponse)
async def upload_file(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    # Save the uploaded file
    file_location = os.path.join("uploads", file.filename)
    with open(file_location, "wb") as f:
        f.write(await file.read())

    # Perform object detection
    detected_objects = detect_objects(file_location)
    
    if not detected_objects:
        raise HTTPException(status_code=404, detail="No cattle detected in the image.")

    # Assuming the first detected object is the one we want to process
    cattle = detected_objects[0]
    breed = predict_breed(cattle['image_path'])
    height = cattle['height']
    width = cattle['width']

    # Calculate weight
    weight = calculate_weight(height, width, breed)

    # Calculate milk yield
    milk_yield = calculate_milk_yield(breed, weight)

    # Prepare response
    response = CattlePredictionResponse(
        breed=breed,
        height=height,
        width=width,
        weight=weight,
        milk_yield=milk_yield
    )

    return JSONResponse(content=response.dict())