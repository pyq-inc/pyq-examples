import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.exceptions import HTTPException
from transformers import ViTFeatureExtractor, ViTForImageClassification
from pydantic import BaseModel
from PIL import Image
import io
import multipart # This is important for reading in files. Do not delete this
import torch

app = FastAPI()
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

# This is useful for testing but feel free to delete it if you want
@app.get("/")
def home():
    return({"Hello":"World"})

@app.post("/predict/")
async def predict(input_sequence: UploadFile = File(...)): # You can also upload lists of files by surrounding the File(...) 
    # Check uploaded file type
    if input_sequence.content_type not in ["image/jpeg", "image/png"]: # Here we are using images but feel free to use any file type
        raise HTTPException(400, detail="Invalid file type")

    # Try to open image file
    try:
        request_object_content = await input_sequence.read()
        image = Image.open(io.BytesIO(request_object_content)) # Again, we're using images here but feel free to explore other options
    except Exception as e:
        return {"response": f"Unable to open the image with error {e}"}

    # Run inference
    try:
        inputs = feature_extractor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits
        # model predicts one of the 1000 ImageNet classes
        predicted_class_idx = logits.argmax(-1).item()
        return({"response":model.config.id2label[predicted_class_idx]})
    except Exception as e:
        return({"response":f"Unable to run inference for this input with error: {e}"})

if __name__ == "__main__":
    uvicorn.run(app, port = 8000, host = "0.0.0.0")

# Special thanks to Jacob Miesner for his help in developing this template