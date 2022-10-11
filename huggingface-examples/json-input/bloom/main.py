from transformers import pipeline, BloomForCausalLM, BloomTokenizerFast, AutoConfig, set_seed
import torch
import os
from accelerate import Accelerator
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
#from torch.profiler import profile, record_function, ProfilerActivity

pwd = os.getcwd() # This is bad practice but whatever. Don't do it again.
app = FastAPI()
tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom")
model = BloomForCausalLM.from_pretrained(pwd)
generator = pipeline(task="text-generation", model=model, tokenizer=tokenizer)

class Item(BaseModel):
    input_sequence: str

@app.get("/")
def home():
	return({"Hello":"World"})

@app.post("/predict/")
async def predict(item: Item):
	input_value = item.input_sequence
	set_seed(42)
	response = generator(input_value, max_length=50, num_return_sequences=1)
	return({"response":response[0]})

# if __name__ == "__main__":
# 	uvicorn.run(app, port = 8080, host = "0.0.0.0")
	#print(generate("There are many animals in a zoo. Some examples include tigers, lions, zebras, "))