from typing import Optional
from fastapi import FastAPI, File, UploadFile
import matplotlib.pyplot as plt
import numpy as np
import uvicorn
import os
from PIL import Image
import sklearn
from joblib import load
import multipart
import io

clf = load('model.joblib')
app = FastAPI()

# Super hacky helper function to transform the input into what the model needs
def prepare_input(image_bytes):
    # Read and prep image
    gray_image = np.array(image_bytes.convert('L')) #you can pass multiple arguments in single line
    Image.fromarray(gray_image).save('gray.png')
    load_img_rz = np.array(Image.open('gray.png').resize((8,8)))
    Image.fromarray(load_img_rz).save('gray88.png')
    img = plt.imread('gray88.png')*255

     # dirty hack to tranform into the format, sklearn needs it
    img = [16-int(round(jj/16)) for j in img for jj in j]
    img = np.array(img)
    #print(img)
    newArray = img.reshape(64)
    newArray = newArray.reshape(1,-1)
    return(newArray)
    
@app.get("/")
def read_root():
	return {"Hello":" World"}

@app.post("/predict/")
async def predict(input_sequence: UploadFile = File(...)):
    try:
        request_object_content = await input_sequence.read()
        image = Image.open(io.BytesIO(request_object_content))
    except Exception as e:
        return {"response": f"Unable to open the image with error {e}"}

    preppedImage = prepare_input(image)
    prediction = clf.predict(preppedImage)
    output = prediction[0]
    return({"response": str(output)})

if __name__ == "__main__": 
    uvicorn.run(app, port = 8000, host="0.0.0.0")