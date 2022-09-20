from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import pickle
import numpy as np
import sklearn

class Item(BaseModel):
    input_sequence: list

app = FastAPI()
model = pickle.load(open('model.sav', 'rb'))

# This is useful for testing but feel free to delete it if you want
@app.get("/")
def home():
	return({"Hello":"World"})

@app.post("/predict/")
async def predict(item: Item):
	input_value = item.input_sequence
	input_transformed = np.array(input_value).reshape(-1, 1).flatten()
	model_output = model.predict(np.array([input_value]))[0]
	return({"response":model_output})

if __name__ == "__main__":
	uvicorn.run(app, port = 8000, host = "0.0.0.0")