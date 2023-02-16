import uvicorn
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.exceptions import HTTPException
from typing import List, Optional, Tuple, Union, TYPE_CHECKING, Iterator, TextIO
import multipart
import io
import numpy as np
import torch
import whisper
import base64
import ffmpeg
import os
import pathlib

app = FastAPI()
model = whisper.load_model("tiny",device="cpu")

def str2bool(string):
    str2val = {"True": True, "False": False}
    if string in str2val:
        return str2val[string]
    else:
        raise ValueError(f"Expected one of {set(str2val.keys())}, got {string}")

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
async def predict(input_sequence: UploadFile = File(...), 
                    verbose: bool = Form(None),
                    temperature: float = Form(0.0),
                    compression_ratio_threshold: float = Form(2.4),
                    logprob_threshold: float = Form(-1.0),
                    no_speech_threshold: float = Form(0.6),
                    condition_on_previous_text: bool = Form(True),
                    best_of: int = Form(5),
                    beam_size: int = Form(5),
                    patience: float = Form(None),
                    length_penalty: float = Form(None),
                    suppress_tokens: str = Form("-1"),
                    initial_prompt: str = Form(None),
                    temperature_increment_on_fallback: float = Form(0.2),
                    ):
    # Check uploaded file type is of correct format
    # if input_sequence.content_type not in ["audio/vnd.wav", "audio/wave", "audio/s-wav", "audio/x-wav", "audio/mpeg", "audio/x-mpeg-2", "audio/x-mpeg", "audio/mpeg3", "audio/mpg", "audio/x-mp3", "audio/x-mpeg3", "audio/x-mpg", "audio/m4a","audio/x-m4a"]:
    #     raise HTTPException(400, detail=f"Invalid file type. Only wav, mp3, mp4, mpeg and m4a files are accepted. Your file was of type {input_sequence.content_type}")

    # Try to open the file
    # try:
    with open(input_sequence.filename, "wb+") as file_object:
        file_object.write(input_sequence.file.read())

    print(temperature)

    if temperature_increment_on_fallback is not None:
        temperature = tuple(np.arange(temperature, 1.0 + 1e-6, temperature_increment_on_fallback))
    else:
        temperature = [temperature]

    print(verbose)
    result = model.transcribe(audio = input_sequence.filename,
                                verbose = verbose,
                                temperature = temperature,
                                compression_ratio_threshold = compression_ratio_threshold,
                                logprob_threshold = logprob_threshold,
                                no_speech_threshold = no_speech_threshold,
                                condition_on_previous_text = condition_on_previous_text,
                                best_of = best_of,
                                beam_size = beam_size,
                                patience = patience,
                                length_penalty = length_penalty,
                                suppress_tokens = suppress_tokens,
                                initial_prompt = initial_prompt)
    return ({"response":result})

# Uncomment the below to test your predict function locally. You can use Postman to post to http://127.0.0.1:8000/predict to get your predictions
if __name__ == "__main__":
    uvicorn.run(app, port = 8000, host = "0.0.0.0")