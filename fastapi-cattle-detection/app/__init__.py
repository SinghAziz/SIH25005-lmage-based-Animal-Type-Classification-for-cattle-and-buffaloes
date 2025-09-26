# app/__init__.py

from fastapi import FastAPI

app = FastAPI()

from .api.routes import detection  # Import routes to register them

app.include_router(detection.router)