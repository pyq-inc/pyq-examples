import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.exceptions import HTTPException
from pydantic import BaseModel
from typing import Iterator, TextIO
import multipart
import io
import numpy as np
import torch
import whisper
import base64
import ffmpeg

app = FastAPI()
model = whisper.load_model("small",device="cpu")

def format_timestamp(seconds: float):
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    return (f"{hours}:" if hours > 0 else "") + f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

def string_vtt(transcript: Iterator[dict]):
    return_val = ''
    for segment in transcript:
        return_val += (
            f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}\n"
            f"{segment['text'].replace('-->', '->')}\n"
            )
    return(return_val)

@app.post("/predict/")
async def predict(input_sequence: UploadFile = File(...)):
    # Check uploaded file type is of correct format
    if input_sequence.content_type not in ["audio/vnd.wav","audio/m4a","audio/mpeg","audio/mp4"]:
        raise HTTPException(400, detail=f"Invalid file type. Only wav, mp3 and m4a files are accepted. Your file was of type {input_sequence.content_type}")

    # Try to open the file
    try:
        with open(input_sequence.filename, "wb+") as file_object:
            file_object.write(input_sequence.file.read())
    except Exception as e:
        return {"response": f"Unable to open the audio file with error {e}"}

    # Run inference
    try:
        result = model.transcribe(input_sequence.filename)
        result_vtt = string_vtt(result["segments"])
        return ({"response":result_vtt})
    except Exception as e:
        return({"response":f"Unable to run inference for this input with error: {e}"})

# Uncomment the below to test your predict function locally. You can use Postman to post to http://127.0.0.1:8000/predict to get your predictions
# if __name__ == "__main__":
#     uvicorn.run(app, port = 8000, host = "0.0.0.0")