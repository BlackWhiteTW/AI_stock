@echo off
chcp 65001
echo ==========================================
echo       正在啟動 AI 股票模擬訓練系統
echo ==========================================

cd /d "%~dp0"

:: 檢查虛擬環境是否存在
if not exist "venv\Scripts\activate.bat" (
    echo [錯誤] 找不到虛擬環境資料夾 venv。
    echo 請確認您是否已在專案目錄下建立了 venv。
    echo.
    pause
    exit
)

echo 正在啟動虛擬環境 (venv)...
call venv\Scripts\activate.bat

echo 正在檢查並安裝必要套件 (若是首次運行會花一點時間)...
pip install -r requirements.txt

echo.
echo 準備啟動 Streamlit 介面...
echo 請稍候，瀏覽器將自動開啟...
echo.

streamlit run app.py

pause