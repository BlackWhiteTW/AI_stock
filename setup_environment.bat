@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ==========================================
echo       AI 股票模擬環境 初始化/修復工具
echo ==========================================

cd /d "%~dp0"

:: --- 檢查 Python ---
echo [1/4] 檢查 Python 環境...
py --version >nul 2>&1
if %errorlevel% equ 0 (
    set "PY_CMD=py -3.12"
    goto :CHECK_VENV
)

python --version >nul 2>&1
if %errorlevel% equ 0 (
    set "PY_CMD=python"
    goto :CHECK_VENV
)

echo [嚴重錯誤] 系統未安裝 Python。請先安裝 Python。
pause
exit /b

:: --- 檢查或建立 venv ---
:CHECK_VENV
if exist "venv\Scripts\activate.bat" (
    echo [2/4] 虛擬環境已存在，跳過建立步驟。
    goto :ACTIVATE_VENV
)

echo [2/4] 正在建立虛擬環境 (venv)...
%PY_CMD% -m venv venv
if %errorlevel% neq 0 (
    echo [錯誤] 無法建立虛擬環境，嘗試使用系統預設 python...
    python -m venv venv
)

:: --- 啟用 venv ---
:ACTIVATE_VENV
echo [3/4] 啟用虛擬環境...
call venv\Scripts\activate.bat

:: --- 安裝/更新套件 ---
:INSTALL_STUFF
echo [4/4] 正在更新 pip 與安裝套件...
python -m pip install --upgrade pip

echo 正在安裝 GPU 版 PyTorch (NVIDIA)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --upgrade

echo 正在安裝其餘依賴套件...
pip install -r requirements.txt

echo.
echo ==========================================
echo       環境配置完成！
echo ==========================================
echo 您現在可以運行 "start_simulation_system.bat" 啟動系統了。
echo.
pause
