# AerialObstacleViT -- CPU-only container image.
#
# The benchmark is deliberately runnable without a GPU: the headline comparison
# is about *frozen foundation features vs open-vocabulary VLMs*, which is a
# data-and-parameter-efficiency question, not a throughput question.  Latency is
# measured separately and honestly recorded in artifacts/results/latency.json.
#
#   docker build -t aerialvit .
#   docker run --rm aerialvit --help
#   docker run --rm -v "$PWD/artifacts:/app/artifacts" aerialvit fetch-data --config configs/smoke.yaml

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    AERIALVIT_DATA_DIR=/app/artifacts/data

# build-essential + libgl1 are needed to compile pycocotools and to satisfy
# OpenCV's shared-library dependencies respectively.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY aerial_vit ./aerial_vit
COPY configs ./configs
COPY scripts ./scripts

# CPU-only torch keeps the image ~10x smaller than the default CUDA wheels.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision \
    && pip install --no-build-isolation -e .

RUN mkdir -p artifacts/results artifacts/runs artifacts/models artifacts/tmp reports/figures

# Fail the build if the CLI is not importable.
RUN aerialvit --help > /dev/null

ENTRYPOINT ["aerialvit"]
CMD ["--help"]
