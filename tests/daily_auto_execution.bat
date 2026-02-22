@echo off
chcp 65001
echo ===================================================
echo   AI 股票自動交易機 - 每日排程 (Daily Routine)
echo   此腳本會執行: 
echo   1. 隨機或指定掃描市場 (更新 ai_shopping_list.csv)
echo   2. 針對購物清單與庫存進行自動買賣決策
echo ===================================================
echo.

cd /d "%~dp0"

echo 正在啟動 Python 機器人核心...
"venv\Scripts\python.exe" -m jobs.daily_job

echo.
echo ===================================================
echo   作業完成。視窗將於 10 秒後自動關閉...
echo ===================================================
timeout /t 10
