@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ==========================================
echo       AI 股票模擬訓練系統 - 快速啟動
echo ==========================================

cd /d "%~dp0"

:: --- 檢查環境 ---
if not exist "venv\Scripts\activate.bat" goto :NO_VENV

:: --- 啟動虛擬環境 ---
echo [1/2] 正在啟動虛擬環境...
call venv\Scripts\activate.bat

:: --- 檢查基本套件 ---
python -c "import streamlit" >nul 2>&1
if %errorlevel% neq 0 goto :FIX_ENV

:: --- 啟動主程式 ---
:RUN_SYSTEM
echo [2/2] 準備啟動主程式...
echo.
echo 提示：如果啟動失敗，請執行 setup_environment.bat 進行完整修復。
echo.

:: 使用 python -m streamlit run 更穩定
python -m streamlit run app.py

if %errorlevel% neq 0 (
    echo.
    echo [錯誤] 程式異常退出 (Exit Code: %errorlevel%)。
    pause
)
exit /b

:: --- 錯誤處理標籤 ---
:NO_VENV
echo [嚴重錯誤] 找不到虛擬環境 (venv)。
echo 請先執行 "setup_environment.bat" 來初始化環境。
echo.
pause
exit /b

:FIX_ENV
echo [警告] 偵測到環境不完整，正在進行快速補救 (安裝 Streamlit)...
python -m pip install streamlit pandas yfinance
if %errorlevel% equ 0 goto :RUN_SYSTEM
echo [錯誤] 自動修復失敗，請手動執行 setup_environment.bat。
pause
exit /b
