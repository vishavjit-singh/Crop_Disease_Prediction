from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
from PIL import Image
from io import BytesIO
import tensorflow as tf
from keras.models import load_model

app = FastAPI()

# Load the main model to predict the crop type
main_model = load_model("model/main_model.h5")

# Load the crop-specific models for disease prediction
crop_models = {
    "Rice": load_model("model/rice.h5"),
    "Wheat": load_model("model/wheat.h5"),
    "Corn": load_model("model/corn.h5")
}

# Define class names for disease prediction
class_names = {
    "Rice": ['Rice_Brown_Spot', 'Rice_Healthy', 'Rice_Leaf_Blast', 'Rice_Neck_Blast'],
    "Wheat": ['White Healthy', 'White Yellow Rus', 'White Red Rus', 'White Green Rus'],
    "Corn": ['Corn_Common_Rust', 'Corn_Gray_Leaf_Spot', 'Corn_Healthy', 'Corn_Northern_Leaf_Blight']
}

@app.get('/ping')
async def ping():
    return 'Hello'

def read_file_as_image(data):
    img = np.array(Image.open(BytesIO(data)))
    return img

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    img = read_file_as_image(await file.read())
    img_batch = np.expand_dims(img, 0)
    
    # Predict the crop type using the main model
    crop_prediction = main_model.predict(img_batch)
    predicted_crop = np.argmax(crop_prediction)
    
    # Use the predicted crop to select the appropriate crop-specific model
    crop_model = crop_models[list(crop_models.keys())[predicted_crop]]
    
    # Predict the disease using the crop-specific model
    disease_prediction = crop_model.predict(img_batch)
    predicted_disease = class_names[list(crop_models.keys())[predicted_crop]][np.argmax(disease_prediction)]
    
    return {
        'crop': list(crop_models.keys())[predicted_crop],
        'disease': predicted_disease,
        'confidence': float(np.max(disease_prediction))
    }

if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=8001)