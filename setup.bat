@echo off
echo ============================================
echo   Credit Receipt App - Setup Script
echo ============================================
echo.

:: Check if Git is installed
git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Git not found. Downloading Git...
    curl -o git_installer.exe -L https://github.com/git-for-windows/git/releases/download/v2.47.1.windows.2/Git-2.47.1.2-64-bit.exe
    echo Installing Git...
    git_installer.exe /VERYSILENT /NORESTART
    echo Git installed. Please CLOSE this terminal and run setup.bat again.
    del git_installer.exe
    pause
    exit /b
)

:: Clone repo if not already inside it
if not exist "app.py" (
    echo Cloning repository...
    git clone https://github.com/krishnakotecha19/credit-receipt-app.git
    cd credit-receipt-app
)

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python not found. Downloading Python 3.10.11...
    echo.
    curl -o python_installer.exe https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe
    echo.
    echo Installing Python 3.10.11...
    echo IMPORTANT: This will install Python with PATH enabled.
    echo.
    python_installer.exe /passive InstallAllUsers=0 PrependPath=1 Include_pip=1
    echo.
    echo Python installed. Please CLOSE this terminal and run setup.bat again.
    echo (PATH needs to refresh)
    del python_installer.exe
    pause
    exit /b
)

echo Python found:
python --version
echo.

:: Create virtual environment
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    echo Virtual environment created.
) else (
    echo Virtual environment already exists.
)
echo.

:: Activate venv and install requirements
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt
echo.

:: Check if .env exists
if not exist ".env" (
    echo Creating .env from template...
    copy .env.example .env
    echo.
    echo !! IMPORTANT: Edit .env file with your database credentials !!
    echo.
)

:: Check if Ollama is installed
ollama --version >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo !! Ollama not found !!
    echo Download from: https://ollama.com/download
    echo After installing, run: ollama pull qwen2.5:3b
) else (
    echo Ollama found:
    ollama --version
    echo Pulling Qwen2.5 3B model...
    ollama pull qwen2.5:3b
)

echo.
echo ============================================
echo   Setup Complete!
echo ============================================
echo.
echo Next steps:
echo   1. Edit .env with your database credentials
echo   2. Make sure Ollama is running: ollama serve
echo   3. Run the app: venv\Scripts\activate ^&^& streamlit run app.py
echo.
pause
