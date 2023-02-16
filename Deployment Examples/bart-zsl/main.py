from transformers import pipeline
from fastapi import FastAPI
import uvicorn
import torch
from pydantic import BaseModel
import json

app = FastAPI()

classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")

class Item(BaseModel):
    input_sequence: str
    candidate_labels: list
    sic_categories: bool

@app.get("/")
def home():
    return({"Hello":"World"})

@app.post("/predict/")
async def predict(item: Item):
    input_sequence = item.input_sequence
    candidate_labels = item.candidate_labels

    if(item.sic_categories == "True" or item.sic_categories == "true" or item.sic_categories):
        with open('sic.json') as user_file:
            parsed_json = json.load(user_file)
            return({"response": classifier(input_sequence, parsed_json["sic_categories"], multi_label=True)})

    print(candidate_labels)
    return({"response": classifier(input_sequence, candidate_labels, multi_label=True)})

if __name__ == "__main__":
    #print(generate_response("testing test is so not"))
    uvicorn.run(app, port = 8080, host = "0.0.0.0")