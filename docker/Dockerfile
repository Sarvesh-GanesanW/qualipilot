FROM python:3.12.13-slim-bookworm@sha256:8a7e7cc04fd3e2bd787f7f24e22d5d119aa590d429b50c95dfe12b3abe52f48b AS builder

ARG EXTRAS=""
ARG UV_VERSION=0.11.21

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /build

RUN python -m pip install --no-cache-dir "uv==${UV_VERSION}"

COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src

RUN if [ -n "${EXTRAS}" ]; then \
        uv sync --locked --no-dev --no-editable --extra "${EXTRAS}"; \
    else \
        uv sync --locked --no-dev --no-editable; \
    fi

FROM python:3.12.13-slim-bookworm@sha256:8a7e7cc04fd3e2bd787f7f24e22d5d119aa590d429b50c95dfe12b3abe52f48b AS runtime

ARG VCS_REF=unknown
ARG VERSION=3.3.0

LABEL org.opencontainers.image.source="https://github.com/Sarvesh-GanesanW/qualipilot" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.version="${VERSION}"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    QUALIPILOT_JSON_LOGS=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}"

RUN groupadd --system --gid 10001 qualipilot \
    && useradd --system --uid 10001 --gid qualipilot qualipilot \
    && mkdir /data /reports /work \
    && chown qualipilot:qualipilot /data /reports /work

COPY --from=builder /opt/venv /opt/venv

USER qualipilot
WORKDIR /work

VOLUME ["/data", "/reports"]

ENTRYPOINT ["qualipilot"]
CMD ["--help"]
