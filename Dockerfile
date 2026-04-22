# Use a Python 3.11 base image
FROM python:3.11-slim

# Install system dependencies for OpenCV and SQLite
# Note: libgl1 replaces the older libgl1-mesa-glx
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install them
# We'll use a single command for the pip installs below
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu \
    flask \
    facenet-pytorch \
    mediapipe \
    opencv-python-headless \
    requests \
    cryptography \
    numpy

# Copy all project files
COPY . .

# Expose the Flask port
EXPOSE 5000

# Run the server
CMD ["python", "server.py"]