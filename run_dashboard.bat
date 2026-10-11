@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\activate.bat (
  echo [ERROR] Virtual environment missing. Run setup_windows.bat first.
  exit /b 1
)
call .venv\Scripts\activate.bat
streamlit run app.py
