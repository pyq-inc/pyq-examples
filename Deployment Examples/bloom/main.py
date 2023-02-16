from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoConfig, set_seed
import torch
import os
from accelerate import Accelerator
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
#from torch.profiler import profile, record_function, ProfilerActivity

app = FastAPI()
checkpoint = "bigscience/bloom-7b1"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype="auto", device_map="auto")
generator = pipeline(task="text-generation", model=model, tokenizer=tokenizer, device=0)

class Item(BaseModel):
    input_sequence: str
    max_new_tokens: int

@app.post("/predict/")
async def predict(item: Item):
	input_value = item.input_sequence
	max_new_tokens = item.max_new_tokens
	set_seed(42)
	response = generator(input_value, max_new_tokens=max_new_tokens, num_return_sequences=1)
	return({"response":response[0]})
	
if __name__ == "__main__":
	uvicorn.run(app, port = 8080, host = "0.0.0.0")