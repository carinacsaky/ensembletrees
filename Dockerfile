FROM python:3.11-slim

WORKDIR /app

# Install system dependencies needed by geopandas/psycopg2
RUN apt-get update && apt-get install -y \
    libgdal-dev \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app and model files
COPY app.py .
COPY output/ ./output/

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
