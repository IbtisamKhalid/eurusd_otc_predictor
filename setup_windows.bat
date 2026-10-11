@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python launcher "py" not found. Install Python 3.11 from https://www.python.org/downloads/windows/
  echo         and tick "Add python.exe to PATH" / "py launcher" during setup.
  exit /b 1
)
if not exist .venv (
  py -3.11 -m venv .venv || py -3 -m venv .venv || (echo [ERROR] Could not create virtual environment & exit /b 1)
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt || (echo [ERROR] Dependency install failed. Check your internet connection / proxy. & exit /b 1)
if not exist .env copy .env.example .env >nul
echo.
echo Setup complete. Next:
echo   1. Edit .env and put your key after OTCHARTS_API_KEY=   (or: $env:OTCHARTS_API_KEY="otc_live_...")
echo   2. python scripts\verify_api.py
echo   3. run_dashboard.bat
endlocal
