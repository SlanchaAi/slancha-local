## slancha-local proxy image — Apache 2.0
## ~600MB final image (debian-slim + Python 3.12 + onnxruntime + libomp + assets)
##
## Build:    docker build -t slancha-local -f docker/Dockerfile .
## Run:      docker run --rm -p 8000:8000 -v ~/.slancha:/root/.slancha slancha-local

FROM python:3.12-slim AS build

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY pyproject.toml README.md NOTICE LICENSE /build/
COPY src /build/src

RUN pip install --upgrade pip && pip install hatchling
RUN python -m hatchling build && ls -la dist/

# ---

FROM python:3.12-slim AS runtime

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SLANCHA_BIND_HOST=0.0.0.0 \
    SLANCHA_BIND_PORT=8000

# libgomp1 = OpenMP runtime, required by treelite for the local classifier
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /build/dist/*.whl /tmp/
RUN pip install /tmp/*.whl && rm /tmp/*.whl

# Optional: if you want the local classifier (default), install the [classifier] extras
RUN pip install treelite scikit-learn

EXPOSE 8000

# Sane defaults: bind 0.0.0.0:8000, classifier=local (zero-egress by default).
# Override with -e SLANCHA_OLLAMA_BASE_URL=http://host.docker.internal:11434 etc.
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=2 \
    CMD python -c "import httpx; httpx.get('http://127.0.0.1:8000/health', timeout=2.0).raise_for_status()" || exit 1

CMD ["slancha-local", "serve", "--host", "0.0.0.0", "--port", "8000"]
