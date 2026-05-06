FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-cloud.txt ./requirements-cloud.txt
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements-cloud.txt

COPY . .

CMD ["python", "scripts/cloud/run_container_workshop.py"]
