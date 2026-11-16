#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="/home/master/coding/21_c_pool/ds_bootcamp/venv/bin/python3"

if [ ! -f "$VENV_PYTHON" ]; then
    VENV_PYTHON="python3"
fi

export PORT="${PORT:-8080}"
echo "Запуск NutriTaste ML веб-сервиса на http://localhost:${PORT}..."
exec "$VENV_PYTHON" "$SCRIPT_DIR/app.py"

