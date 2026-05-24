@echo off
echo Creating virtual environment...

REM Create venv
python -m venv .venv

IF %ERRORLEVEL% NEQ 0 (
    echo Failed to create virtual environment.
    exit /b 1
)

echo Activating virtual environment...

REM Activate venv
call .venv\Scripts\activate

echo Virtual environment activated.
echo Installing dependencies (if requirements.txt exists)...

IF EXIST requirements.txt (
    pip install -r requirements.txt
) ELSE (
    echo No requirements.txt found.
)

echo Done.
cmd /k