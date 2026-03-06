# 每日新聞自動收集 PowerShell 腳本
# 可加入 Windows 工作排程器中每天早上執行

Write-Host "====================================" -ForegroundColor Cyan
Write-Host "每日新聞自動收集任務" -ForegroundColor Cyan
Write-Host "====================================" -ForegroundColor Cyan
Write-Host ""

# 切換到專案目錄
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptPath
Set-Location $projectRoot

Write-Host "專案目錄: $projectRoot" -ForegroundColor Yellow

# 啟動虛擬環境（如果有的話）
$venvActivate = Join-Path $projectRoot "venv\Scripts\Activate.ps1"
if (Test-Path $venvActivate) {
    Write-Host "啟動虛擬環境..." -ForegroundColor Yellow
    & $venvActivate
}

# 執行新聞收集
Write-Host "開始收集新聞..." -ForegroundColor Green
python jobs\daily_news_collector.py collect

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✅ 任務完成！" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "❌ 任務執行失敗！" -ForegroundColor Red
}

Write-Host ""

# 可選：暫停以查看結果
# Read-Host "按 Enter 鍵繼續..."
