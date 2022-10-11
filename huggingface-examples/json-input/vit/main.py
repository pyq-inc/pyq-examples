from fastapi import FastAPI
import uvicorn
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import requests
import torch
from pydantic import BaseModel

class Item(BaseModel):
    input_sequence: str

app = FastAPI()
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

@app.get("/")
def home():
	return({"Hello":"World"})

@app.post("/predict/")
async def predict(item: Item):
	url = item.input_sequence
	image = Image.open(requests.get(url, stream=True).raw)
	inputs = feature_extractor(images=image, return_tensors="pt")
	outputs = model(**inputs)
	logits = outputs.logits
	# model predicts one of the 1000 ImageNet classes
	predicted_class_idx = logits.argmax(-1).item()
	return({"response":model.config.id2label[predicted_class_idx]})

#if __name__ == "__main__":
	#uvicorn.run(app, port = 8080, host = "0.0.0.0")
	#print(generate("There are many animals in a zoo. Some examples include tigers, lions, zebras, "))