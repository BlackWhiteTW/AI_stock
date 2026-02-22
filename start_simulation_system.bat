@echo off
chcp 65001
echo ==========================================
echo       正在啟動 AI 股票模擬訓練系統
echo ==========================================

cd /d "%~dp0"

:: ------------------------------------------------
:: 階段 1: 檢查環境是否存在
:: ------------------------------------------------
if exist "venv\Scripts\activate.bat" (
    goto :START_SYSTEM
)

:: ------------------------------------------------
:: 階段 2: 若環境不存在，進行建立 (只在沒環境時執行)
:: ------------------------------------------------
:CREATE_ENV
echo.
echo [初始化] 未檢測到虛擬環境，正在為您建立...
echo ------------------------------------------

:: 優先嘗試指定 Python 3.12 (支援 GPU 加速的最佳版本)
echo 嘗試尋找 Python 3.12...
py -3.12 -m venv venv
if errorlevel 1 (
    echo [提示] 找不到 Python 3.12，嘗試使用系統預設 Python...
    py -m venv venv
)
if errorlevel 1 (
    echo [重試] 'py' 指令無回應，嘗試直接使用 'python'...
    python -m venv venv
)

:: 確認建立結果
if not exist "venv\Scripts\activate.bat" (
    echo.
    echo [嚴重錯誤] 無法建立環境！
    echo 請確認您已安裝 Python 並且勾選 'Add to PATH'。
    echo.
    pause
    exit
)

echo 環境建立成功！
echo 正在升級 pip...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
goto :INSTALL_DEPS

:: ------------------------------------------------
:: 階段 3: 啟動系統
:: ------------------------------------------------
:START_SYSTEM
echo 正在啟動虛擬環境...
call venv\Scripts\activate.bat

:INSTALL_DEPS
:: 檢查並安裝套件
echo.
echo [優化] 正在強制安裝支援 NVIDIA 顯卡的 PyTorch 版本...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --upgrade
if errorlevel 1 (
    echo [警告] GPU 版 PyTorch 安裝可能有誤，將嘗試標準安裝...
)

echo 正在檢查其他套件依賴...
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [警告] 套件安裝發生錯誤，將嘗試繼續執行...
)

echo.
echo ==========================================
echo       準備啟動 Streamlit 介面
echo ==========================================
echo.
echo 請稍候，瀏覽器將自動開啟...
echo.

streamlit run app.py

pause