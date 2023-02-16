FROM python:3.10
COPY . .

RUN pip3 install -r requirements.txt && pip3 install python-multipart && pip3 install uvicorn

EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]

