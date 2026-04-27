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
    QUALIPILOT_JSON_LOGS=1

RUN groupadd --system --gid 1000 qualipilot \
    && useradd  --system --uid 1000 --gid qualipilot qualipilot \
    && mkdir /data /reports \
    && chown qualipilot:qualipilot /data /reports

COPY --from=builder /usr/local/lib/python3.12/site-packages \
     /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin/qualipilot /usr/local/bin/qualipilot

USER qualipilot
WORKDIR /work

# workspace layout is opinionated: mount inputs on /data, reports land
# in /reports, so compose files stay simple
VOLUME ["/data", "/reports"]

ENTRYPOINT ["qualipilot"]
CMD ["--help"]
