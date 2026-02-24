# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies for psycopg2 and other tools
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container
COPY . .

# Render injects PORT env var; fall back to 5002 locally
ENV PORT=5002
EXPOSE $PORT

# Run using gunicorn on the dynamic port
CMD gunicorn --bind "0.0.0.0:$PORT" --workers 2 --threads 4 app:app
