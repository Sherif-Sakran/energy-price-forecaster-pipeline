# Use an official Python runtime as a parent image
FROM python:3.11-slim

# --- THE FIX: Install OpenMP library for LightGBM ---
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*
    
# Set the working directory in the container
WORKDIR /code

# HF Spaces requirement: Create a non-root user with UID 1000
RUN useradd -m -u 1000 user

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY ./app /code/app

# Change ownership of the /code directory to the non-root user
RUN chown -R user:user /code

# Switch to the non-root user
USER user

# HF Spaces routes external traffic to port 7860 by default
EXPOSE 7860

# Command to run the FastAPI application using Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]