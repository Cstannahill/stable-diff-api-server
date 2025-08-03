FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y git && \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY app /app/app

CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]
