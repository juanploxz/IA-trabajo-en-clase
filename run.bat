@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

set "VENV_PY=.venv\Scripts\python.exe"
if exist "%VENV_PY%" goto install

set "PYTHON_EXE="
set "PYTHON_ARGS="
py -3 -c "import sys; raise SystemExit(sys.version_info < (3, 10))" >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_EXE=py"
    set "PYTHON_ARGS=-3"
    goto create_venv
)
python -c "import sys; raise SystemExit(sys.version_info < (3, 10))" >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_EXE=python"
    goto create_venv
)
python3 -c "import sys; raise SystemExit(sys.version_info < (3, 10))" >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_EXE=python3"
    goto create_venv
)

echo ERROR: No se encontro Python 3.10 o superior.
pause
exit /b 1

:create_venv
echo Creando el entorno virtual con %PYTHON_EXE% %PYTHON_ARGS%...
%PYTHON_EXE% %PYTHON_ARGS% -m venv .venv
if errorlevel 1 goto error

:install
echo Instalando dependencias...
"%VENV_PY%" -m pip install -r requirements.txt
if errorlevel 1 goto error

echo Ejecutando GridWorld...
"%VENV_PY%" main.py %*
if errorlevel 1 goto error
exit /b 0

:error
echo.
echo La ejecucion termino con un error. Revise el mensaje anterior.
pause
exit /b 1
