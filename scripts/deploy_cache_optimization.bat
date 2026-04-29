@echo off
REM 快取優化部署腳本
echo ================================================
echo     快取系統優化部署
echo ================================================
echo.

REM 1. 執行快取遷移
echo [1/3] 開始遷移舊快取到統一系統...
python scripts\migrate_to_unified_cache.py
if errorlevel 1 (
    echo 遷移失敗！
    pause
    exit /b 1
)

echo.
echo [2/3] 執行效能測試...
python tests\test_cache_performance.py

echo.
echo [3/3] 清理過期快取...
python -c "from services.cache_manager import get_cache_manager; cm = get_cache_manager(); print(f'清理了 {cm.cleanup_expired()} 個過期項目')"

echo.
echo ================================================
echo     部署完成！
echo ================================================
echo.
echo 建議：
echo 1. 查看效能提升報告（上方顯示）
echo 2. 更新程式碼以使用新的 CacheManager
echo 3. 定期執行 cleanup_expired() 清理過期資料
echo.
pause
