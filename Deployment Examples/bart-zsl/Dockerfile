FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

COPY . .

RUN pip3 install -r requirements.txt

EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]