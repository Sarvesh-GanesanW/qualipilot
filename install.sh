#!/usr/bin/env bash
# One-click installer for datapilot on macOS/Linux.
#
# usage:
#     ./install.sh                # core only
#     ./install.sh --bedrock      # adds boto3
#     ./install.sh --all          # adds every optional dep
#     ./install.sh --dev          # editable + dev + all extras
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

EXTRAS="core"
for arg in "$@"; do
    case "$arg" in
        --bedrock)  EXTRAS="bedrock" ;;
        --ollama)   EXTRAS="ollama" ;;
        --openai)   EXTRAS="openai" ;;
        --dask)     EXTRAS="dask" ;;
        --all)      EXTRAS="all" ;;
        --dev)      EXTRAS="dev" ;;
        -h|--help)
            grep "^#" "$0" | cut -c3-
            exit 0
            ;;
        *)
            echo "unknown flag: $arg"; exit 2
            ;;
    esac
done

# prefer uv when present; falls back to pip in a venv
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "python3 not found; install Python 3.11+ first"
    exit 1
fi

VENV_DIR="${VENV_DIR:-.venv}"
if [[ ! -d "${VENV_DIR}" ]]; then
    echo "creating venv at ${VENV_DIR}"
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip >/dev/null

if command -v uv >/dev/null 2>&1; then
    INSTALLER=(uv pip install)
else
    echo "uv not found; falling back to pip (slower)"
    INSTALLER=(python -m pip install)
fi

case "${EXTRAS}" in
    core)
        "${INSTALLER[@]}" -e .
        ;;
    dev)
        "${INSTALLER[@]}" -e ".[dev,all]"
        pre-commit install || true
        ;;
    *)
        "${INSTALLER[@]}" -e ".[${EXTRAS}]"
        ;;
esac

echo
echo "installed. activate your shell with:"
echo "    source ${VENV_DIR}/bin/activate"
echo "then try:"
echo "    datapilot --help"
