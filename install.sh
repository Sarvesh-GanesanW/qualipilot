#!/usr/bin/env bash
# Install qualipilot from this checkout on macOS or Linux.
#
# usage:
#     ./install.sh
#     ./install.sh --bedrock
#     ./install.sh --linking --duckdb
#     ./install.sh --all
#     ./install.sh --dev
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

EXTRAS=()
DEV_MODE=0
for arg in "$@"; do
    case "${arg}" in
        --bedrock) EXTRAS+=("bedrock") ;;
        --dask) EXTRAS+=("dask") ;;
        --ollama) EXTRAS+=("ollama") ;;
        --openai) EXTRAS+=("openai") ;;
        --linking) EXTRAS+=("linking") ;;
        --duckdb) EXTRAS+=("duckdb") ;;
        --spark) EXTRAS+=("spark") ;;
        --all) EXTRAS=("all") ;;
        --dev) DEV_MODE=1 ;;
        -h | --help)
            sed -n 's/^# \{0,1\}//p' "$0"
            exit 0
            ;;
        *)
            echo "unknown flag: ${arg}" >&2
            exit 2
            ;;
    esac
done

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "${PYTHON_BIN} not found; install Python 3.11-3.13 first" >&2
    exit 1
fi
if ! "${PYTHON_BIN}" -c \
    'import sys; raise SystemExit(not ((3, 11) <= sys.version_info < (3, 14)))'
then
    echo "qualipilot requires Python 3.11-3.13" >&2
    exit 1
fi

VENV_DIR="${VENV_DIR:-.venv}"
if [[ ! -d "${VENV_DIR}" ]]; then
    echo "creating virtual environment at ${VENV_DIR}"
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

if ! python -c \
    'import sys; raise SystemExit(not ((3, 11) <= sys.version_info < (3, 14)))'
then
    echo "${VENV_DIR} uses an unsupported Python; recreate it" >&2
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found; installing uv 0.11.21"
    python -m pip install "uv==0.11.21"
fi

SYNC_ARGS=(sync --locked --active)
if ((!DEV_MODE)); then
    SYNC_ARGS+=(--no-dev --no-editable)
fi
for extra in "${EXTRAS[@]}"; do
    SYNC_ARGS+=(--extra "${extra}")
done
uv "${SYNC_ARGS[@]}"

if ((DEV_MODE)); then
    pre-commit install
fi

echo
echo "installed. activate the environment with:"
echo "    source ${VENV_DIR}/bin/activate"
echo "then run:"
echo "    qualipilot --help"
