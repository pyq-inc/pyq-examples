from typing import Optional
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    input_sequence: str

@app.get("/")
def sayHello():
	return{"response": "fuck this shit im out"}

@app.post("/predict/")
async def predict(item: Item):
	return({"response": item.input_sequence})

def testFunction():
	raise Exception("yoooo")

if __name__ == "__main__":
	uvicorn.run(app, port = 8080, host="0.0.0.0")
