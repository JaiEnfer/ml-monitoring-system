FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src ./src

# Create folders expected by the app
RUN mkdir -p artifacts reports

# Train model at build-time so the container is ready to serve immediately
# (Good for portfolio demo; in production you might do this in a separate training pipeline.)
RUN python -m src.train

EXPOSE 8000

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
