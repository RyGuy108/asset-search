FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
# CPU-only torch keeps the image ~2 GB smaller than the default CUDA wheel
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "assetsearch.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
