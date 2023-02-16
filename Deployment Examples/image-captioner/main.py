
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch
from PIL import Image
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile, Form
import io

app = FastAPI()

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

@app.post("/predict/")
async def predict(input_sequence: UploadFile = File(...)):
    max_length = 16
    num_beams = 4
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

    images = []
    request_object_content = await input_sequence.read()
    i_image = Image.open(io.BytesIO(request_object_content))
    if i_image.mode != "RGB":
        i_image = i_image.convert(mode="RGB")

    images.append(i_image)

    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return {"response": preds}

if __name__ == "__main__":
    uvicorn.run(app, port = 8080, host = "0.0.0.0")