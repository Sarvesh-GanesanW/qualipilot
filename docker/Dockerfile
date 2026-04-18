FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

# uv is ~10x faster than pip and pins reproducibly
RUN pip install --no-cache-dir uv==0.4.18

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

# default image ships the bedrock + dask extras; local users can
# build with --build-arg EXTRAS=ollama to shrink the footprint
ARG EXTRAS=bedrock
RUN uv pip install --system --no-cache --compile-bytecode \
    ".[${EXTRAS}]"

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DATAPILOT_JSON_LOGS=1

RUN groupadd --system --gid 1000 datapilot \
    && useradd  --system --uid 1000 --gid datapilot datapilot \
    && mkdir /data /reports \
    && chown datapilot:datapilot /data /reports

COPY --from=builder /usr/local/lib/python3.12/site-packages \
     /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin/datapilot /usr/local/bin/datapilot

USER datapilot
WORKDIR /work

# workspace layout is opinionated: mount inputs on /data, reports land
# in /reports, so compose files stay simple
VOLUME ["/data", "/reports"]

ENTRYPOINT ["datapilot"]
CMD ["--help"]
