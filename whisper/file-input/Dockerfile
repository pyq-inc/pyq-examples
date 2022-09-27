FROM python:3.9.14-bullseye 
COPY . .

RUN pip3 install -r requirements.txt && pip3 install python-multipart && pip3 install uvicorn && apt-get update && apt-get install -y \
    ffmpeg \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/* && whisper --model small dummy.wav; exit 0

EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
