FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_PORT=1982 \
    DATA_DIR=/app/data \
    OUTPUT_DIR=/app/outputs

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY project.py nicegui_app.py README.md ./
COPY Smart_Demand_Signals_Analysis.ipynb ./

RUN mkdir -p /app/data /app/outputs

EXPOSE 1982

CMD ["python", "nicegui_app.py"]
