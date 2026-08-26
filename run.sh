#!/bin/sh
set -u

cd -- "$(dirname -- "$0")" || exit 1
VENV_PY=".venv/bin/python"

if [ ! -x "$VENV_PY" ]; then
    PYTHON_EXE=""
    for candidato in python3 python; do
        if command -v "$candidato" >/dev/null 2>&1 && \
            "$candidato" -c 'import sys; raise SystemExit(sys.version_info < (3, 10))' \
            >/dev/null 2>&1; then
            PYTHON_EXE="$candidato"
            break
        fi
    done

    if [ -z "$PYTHON_EXE" ]; then
        echo "ERROR: No se encontró Python 3.10 o superior." >&2
        exit 1
    fi

    echo "Creando el entorno virtual con $PYTHON_EXE..."
    "$PYTHON_EXE" -m venv .venv || exit 1
fi

echo "Instalando dependencias..."
"$VENV_PY" -m pip install -r requirements.txt || exit 1

echo "Ejecutando GridWorld..."
"$VENV_PY" main.py "$@"
estado=$?
if [ "$estado" -ne 0 ]; then
    echo "La ejecución terminó con un error. Revise el mensaje anterior." >&2
    if [ -t 0 ]; then
        printf "Presione Enter para cerrar..."
        read -r _respuesta
    fi
fi
exit "$estado"
