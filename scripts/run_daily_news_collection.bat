@echo off
REM 每日新聞自動收集批次檔
REM 可放入 Windows 工作排程器中每天早上執行

echo ====================================
echo 每日新聞自動收集任務
echo ====================================
echo.

REM 切換到專案目錄
cd /d "%~dp0.."

REM 啟動虛擬環境（如果有的話）
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

REM 執行新聞收集
echo 開始收集新聞...
python jobs\daily_news_collector.py collect

echo.
echo 任務完成！
echo.

REM 暫停以查看結果（可選）
REM pause
