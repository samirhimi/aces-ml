FROM python:3.9-slim

WORKDIR /app

# Install build essentials and security updates
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create requirements.txt file with necessary packages
COPY requirements.txt ./

# Install Python packages and remove build dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    apt-get purge -y build-essential && \
    apt-get autoremove -y

# Create necessary directories
RUN mkdir -p /app/data /app/models /app/csv_backups

# Copy the application code
COPY ML.py .

# Set default environment variables
ENV MIN_SAMPLES_FOR_TRAINING=1 \
    BATCH_SIZE=5 \
    BATCH_TIMEOUT=1.0 \
    PYTHONUNBUFFERED=1 \
    FLASK_ENV=production \
    FLASK_APP=ML.py

# Add healthcheck with longer start period
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import os; exit(0 if os.path.exists('models/random_forest_model.joblib') else 1)"

# Command to run the ML script
CMD ["python", "ML.py"]