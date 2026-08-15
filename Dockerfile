FROM python:3.11-slim-bookworm

ARG INSTALL_INFERENCE=false

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    INSTALL_INFERENCE=${INSTALL_INFERENCE} \
    HOST=0.0.0.0 \
    PORT=4173

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --system appuser \
    && useradd --system --gid appuser --create-home --home-dir /home/appuser appuser

COPY requirements.txt /app/requirements.txt
COPY requirements-inference.txt /app/requirements-inference.txt

RUN python -m pip install --upgrade pip \
    && python -m pip install -r /app/requirements.txt \
    && if [ "$INSTALL_INFERENCE" = "true" ]; then python -m pip install -r /app/requirements-inference.txt; fi

COPY app /app/app
COPY backend /app/backend
COPY resources /app/resources
COPY runtime_src /app/runtime_src
COPY server.py /app/server.py
COPY README.md /app/README.md

RUN mkdir -p /app/runtime_static/uploads /app/runtime_static/results \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 4173

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=5 \
  CMD python -c "import sys, urllib.request; urllib.request.urlopen('http://127.0.0.1:4173/health', timeout=3); sys.exit(0)"

CMD ["gunicorn", "--bind", "0.0.0.0:4173", "--workers", "1", "--threads", "4", "--timeout", "180", "--access-logfile", "-", "--error-logfile", "-", "backend.api:app"]
