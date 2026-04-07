# ─── FastAPI Backend Dockerfile ───────────────────────────────────────────────
# Base image: Python 3.11 slim (matches requirements.txt pin comment)
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Install Python dependencies first (layer-cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source, models, and data
COPY src/ ./src/
COPY models/ ./models/
COPY data/ ./data/

# Expose the port uvicorn will listen on
EXPOSE 8000

# Run the FastAPI app as a package (src.api) so relative imports work
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
