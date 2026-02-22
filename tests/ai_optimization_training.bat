@echo off
chcp 65001 >nul

REM Switch to script directory
cd /d "%~dp0"

echo ========================================
echo AI Optimization Training System v4.0
echo ========================================
echo.

echo Select Training Mode:
echo.
echo [1] Single Stock Optimization (Hyperparameter Tuning)
echo [2] Model Comparison (LSTM vs GRU)
echo [3] Batch Optimization (Multiple Stocks)
echo [4] Quick Test (Taiwan Semiconductor 2330)
echo.

set /p choice="Please select (1-4): "

if "%choice%"=="1" (
    set /p ticker="Enter stock ticker (e.g., 2330.TW): "
    set /p name="Enter stock name: "
    echo.
    echo Starting hyperparameter optimization...
    python scripts\advanced_trainer.py --mode optimize --ticker %ticker% --name %name%
) else if "%choice%"=="2" (
    set /p ticker="Enter stock ticker (e.g., 2330.TW): "
    set /p name="Enter stock name: "
    echo.
    echo Starting model comparison...
    python scripts\advanced_trainer.py --mode compare --ticker %ticker% --name %name%
) else if "%choice%"=="3" (
    set /p max="Max stocks to train (empty for all): "
    echo.
    echo Starting batch optimization...
    if "%max%"=="" (
        python scripts\advanced_trainer.py --mode batch
    ) else (
        python scripts\advanced_trainer.py --mode batch --max %max%
    )
) else if "%choice%"=="4" (
    echo.
    echo Starting quick test on Taiwan Semiconductor...
    python scripts\advanced_trainer.py --mode optimize --ticker 2330.TW --name 台積電
) else (
    echo Invalid choice
    pause
    exit /b 1
)

echo.
echo ========================================
echo Training completed!
echo Check data/optimization_report_*.txt for results
echo ========================================
pause
