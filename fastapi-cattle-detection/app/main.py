from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from app.api.routes.detection import router as detection_router

app = FastAPI()

app.include_router(detection_router)

@app.get("/")
async def root():
    return {"message": "Welcome to the Cattle Detection API!"}