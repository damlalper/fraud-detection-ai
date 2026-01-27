@echo off
REM Project Setup Script for Fraud Detection System (Windows)

echo ==========================================
echo Setting up Fraud Detection System
echo ==========================================

REM Check Python version
echo Checking Python version...
python --version

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo Installing Python dependencies...
pip install -r requirements.txt

REM Install development dependencies
echo Installing development dependencies...
pip install black flake8 mypy pre-commit pytest pytest-cov

REM Install package in development mode
echo Installing package in development mode...
pip install -e .

REM Setup pre-commit hooks
echo Setting up pre-commit hooks...
pre-commit install

REM Create .env file from example
if not exist .env (
    echo Creating .env file...
    copy .env.example .env
    echo Please update .env file with your API keys
)

REM Create necessary directories
echo Creating project directories...
if not exist data\raw mkdir data\raw
if not exist data\processed mkdir data\processed
if not exist data\policies mkdir data\policies
if not exist models mkdir models
if not exist logs mkdir logs
if not exist notebooks mkdir notebooks
if not exist tests\unit mkdir tests\unit
if not exist tests\integration mkdir tests\integration
if not exist tests\load mkdir tests\load

REM Initialize MLflow
echo Initializing MLflow...
if not exist mlruns mkdir mlruns
if not exist mlartifacts mkdir mlartifacts

REM Download sample data
echo Downloading datasets...
python scripts\download_data.py

echo ==========================================
echo Setup completed successfully!
echo ==========================================
echo.
echo Next steps:
echo 1. Activate virtual environment: venv\Scripts\activate
echo 2. Update .env file with your API keys
echo 3. Run the API: uvicorn src.api.main:app --reload
echo 4. Visit http://localhost:8000/docs for API documentation
echo.
pause
