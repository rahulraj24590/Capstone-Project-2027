# LandslideGuard — Production-Style Capstone Container
# Runs the Streamlit 9-page dashboard on port 8501 (or override CMD for FastAPI port 8000).
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

WORKDIR /app

# Install runtime dependencies first for Docker layer caching
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy project source and configuration
COPY . .

# Prepare runtime directories, initialize empty SQLite schema,
# generate pipeline-test synthetic data, and train model artifacts locally inside container
RUN mkdir -p data/raw data/processed models logs .cache \
    && python -c "from src.database.db import initialize_database; initialize_database()" \
    && python scripts/prepare_data.py \
    && python -m src.models.train

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=5)" || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]

