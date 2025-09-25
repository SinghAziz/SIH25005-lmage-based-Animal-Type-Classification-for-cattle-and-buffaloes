from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
from torchvision import models, transforms
import torch.nn as nn
from PIL import Image
import io, os


app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#-- CLASS NAMES --#
class_names = [
    'Alambadi', 'Amritmahal', 'Ayrshire', 'Banni', 'Bargur',
    'Bhadawari', 'Brown_Swiss', 'Dangi', 'Deoni', 'Gir',
    'Guernsey', 'Hallikar', 'Hariana', 'Holstein_Friesian',
    'Jaffrabadi', 'Jersey', 'Kangayam', 'Kankrej', 'Kasargod',
    'Kenkatha', 'Kherigarh', 'Khillari', 'Krishna_Valley',
    'Malnad_gidda', 'Mehsana', 'Murrah', 'Nagori', 'Nagpuri',
    'Nili_Ravi', 'Nimari', 'Ongole', 'Pulikulam', 'Rathi',
    'Red_Dane', 'Red_Sindhi', 'Sahiwal', 'Surti', 'Tharparkar',
    'Toda', 'Umblachery', 'Vechur'
]

animal_map = {
    'Alambadi': 'Cattle', 'Amritmahal': 'Cattle', 'Ayrshire': 'Cattle',
    'Banni': 'Buffalo', 'Bargur': 'Cattle', 'Bhadawari': 'Buffalo',
    'Brown_Swiss': 'Cattle', 'Dangi': 'Cattle', 'Deoni': 'Cattle',
    'Gir': 'Cattle', 'Guernsey': 'Cattle', 'Hallikar': 'Cattle',
    'Hariana': 'Cattle', 'Holstein_Friesian': 'Cattle', 'Jaffrabadi': 'Buffalo',
    'Jersey': 'Cattle', 'Kangayam': 'Cattle', 'Kankrej': 'Cattle',
    'Kasargod': 'Cattle', 'Kenkatha': 'Cattle', 'Kherigarh': 'Cattle',
    'Khillari': 'Cattle', 'Krishna_Valley': 'Cattle', 'Malnad_gidda': 'Cattle',
    'Mehsana': 'Buffalo', 'Murrah': 'Buffalo', 'Nagori': 'Cattle',
    'Nagpuri': 'Buffalo', 'Nili_Ravi': 'Buffalo', 'Nimari': 'Cattle',
    'Ongole': 'Cattle', 'Pulikulam': 'Cattle', 'Rathi': 'Cattle',
    'Red_Dane': 'Cattle', 'Red_Sindhi': 'Cattle', 'Sahiwal': 'Cattle',
    'Surti': 'Buffalo', 'Tharparkar': 'Cattle', 'Toda': 'Cattle',
    'Umblachery': 'Cattle', 'Vechur': 'Cattle'
}

#--Model loading function--#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

num_classes = 41  # Must match your checkpoint
model_path = os.path.join(BASE_DIR, "Model", "best_model.pth")

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

#--Image transforms--#
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

#-- Prediction function --#

def predict_image(model, image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, preds = torch.max(outputs,1)
    breed = class_names[preds.item()]
    animal = animal_map.get(breed, "Unknown")
    return {"animal": animal, "breed": breed}


#--FastAPI APP--#

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "Model", "best_model.pth")

def load_model(model_path, num_classes):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model(model_path, len(class_names))

@app.get("/")
def root():
         return {"message": "Welcome to the Cattle and Buffalo Breed Prediction API!"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))  # For Render
    uvicorn.run("Backend.main:app", host="0.0.0.0", port=port)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
     contents = await file.read()
     result = predict_image(model, contents)
     return JSONResponse(content=result)



