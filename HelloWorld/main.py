from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel

class Item(BaseModel):
    input_sequence: str

app = FastAPI()

@app.get("/")
def read_root():
	return {"Hello":" World"}

@app.post("/predict/")
async def predict(item: Item):
	input_value = item.input_sequence
	return({"response":f"Hello {input_value} World"})

if __name__ == "__main__":
	uvicorn.run(app, port = 8000, host = "0.0.0.0")
