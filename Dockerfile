FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# optional: pre-download the rembg model (fallback for non-flat backgrounds); build continues if offline
RUN python -c "from rembg import new_session; new_session('isnet-general-use')" || true
COPY app.py .
ENV DATA_DIR=/data/tshirt-generator
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "1800"]
