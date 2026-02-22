@echo off
chcp 65001 >nul

REM 切換到腳本所在目錄
cd /d "%~dp0"

echo ========================================
echo Market Training System
echo ========================================
echo.

echo Select Training Mode:
echo.
echo [1] Quick Training (50 important stocks)
echo [2] Full Training (all stocks needing update)
echo [3] Force Training (ignore time checks)
echo [4] Untrained Stocks Only
echo.

set /p choice="Please select (1-4): "

if "%choice%"=="1" (
    echo.
    echo Starting quick training...
    python jobs\market_trainer.py --max 50 --priority important
) else if "%choice%"=="2" (
    echo.
    echo Starting full training...
    python jobs\market_trainer.py --priority important
) else if "%choice%"=="3" (
    echo.
    echo Starting force training...
    python jobs\market_trainer.py --priority all --force
) else if "%choice%"=="4" (
    echo.
    echo Starting untrained stocks training...
    python jobs\market_trainer.py --priority untrained
) else (
    echo Invalid choice
    pause
    exit /b 1
)

echo.
echo ========================================
echo Training completed!
echo ========================================
pause
