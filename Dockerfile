FROM python:3.10-slim

# Install minimal system dependencies required by opencv-python-headless
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all code
COPY . .

# Hugging Face Spaces Docker environments map port 7860 by default
EXPOSE 7860

# Run the FastAPI app on port 7860
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
