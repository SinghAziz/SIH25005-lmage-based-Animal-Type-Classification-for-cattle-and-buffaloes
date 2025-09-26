# FastAPI Cattle Detection

This project is a FastAPI application designed for cattle detection and prediction of various parameters such as breed, weight, and milk yield based on uploaded images. The application utilizes machine learning models for object detection and predictions.

## Project Structure

```
fastapi-cattle-detection
├── app
│   ├── __init__.py
│   ├── main.py
│   ├── api
│   │   ├── __init__.py
│   │   └── routes
│   │       ├── __init__.py
│   │       └── detection.py
│   ├── core
│   │   ├── __init__.py
│   │   └── config.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── breed_predictor.py
│   │   ├── object_detector.py
│   │   ├── weight_predictor.py
│   │   └── milk_predictor.py
│   ├── schemas
│   │   ├── __init__.py
│   │   └── cattle.py
│   ├── services
│   │   ├── __init__.py
│   │   ├── detection_service.py
│   │   ├── breed_service.py
│   │   ├── weight_service.py
│   │   └── milk_service.py
│   └── utils
│       ├── __init__.py
│       ├── image_processing.py
│       └── model_loader.py
├── saved_models
│   ├── best_model.pth
│   ├── weight_predictor_model.pth
│   ├── milk_yield_model.pth
│   ├── breed_encoder.pkl
│   ├── scaler_X.pkl
│   ├── scaler_y.pkl
│   └── milk_breed_encoder.pkl
├── uploads
├── requirements.txt
├── .env
└── README.md
```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd fastapi-cattle-detection
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Set up environment variables in the `.env` file as needed.

## Usage

1. Start the FastAPI application:
   ```
   uvicorn app.main:app --reload
   ```

2. Access the API documentation at `http://127.0.0.1:8000/docs`.

3. Use the `/upload` endpoint to upload images for cattle detection and predictions.

## Features

- Object detection for cattle in images.
- Breed prediction based on detected cattle.
- Calculation of weight based on height and width parameters.
- Prediction of milk yield based on breed and weight.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.