# 設置 Windows 工作排程器任務
# 自動在每天早上 7:30 執行新聞收集

$TaskName = "AI_Stock_Daily_News_Collection"
$Description = "每天自動收集市場新聞並存入周檔案，供 AI 模型訓練使用"

# 獲取當前腳本所在目錄的上一層（專案根目錄）
$ScriptPath = Split-Path -Parent $PSScriptRoot
$PythonScript = Join-Path $ScriptPath "jobs\daily_news_collector.py"
$LogPath = Join-Path $ScriptPath "logs\news_collection.log"

# 確保 logs 目錄存在
$LogDir = Join-Path $ScriptPath "logs"
if (!(Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

Write-Host "====================================" -ForegroundColor Cyan
Write-Host "設置每日新聞收集排程任務" -ForegroundColor Cyan
Write-Host "====================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "專案路徑: $ScriptPath" -ForegroundColor Yellow
Write-Host "Python 腳本: $PythonScript" -ForegroundColor Yellow
Write-Host "日誌路徑: $LogPath" -ForegroundColor Yellow
Write-Host ""

# 檢查 Python 腳本是否存在
if (!(Test-Path $PythonScript)) {
    Write-Host "❌ 找不到 Python 腳本: $PythonScript" -ForegroundColor Red
    Write-Host "請確保在正確的目錄執行此腳本" -ForegroundColor Red
    exit 1
}

# 找到 Python 執行檔
$PythonExe = "python"
$venvPython = Join-Path $ScriptPath "venv\Scripts\python.exe"
if (Test-Path $venvPython) {
    $PythonExe = $venvPython
    Write-Host "使用虛擬環境 Python: $PythonExe" -ForegroundColor Green
} else {
    Write-Host "使用系統 Python: $PythonExe" -ForegroundColor Yellow
}

# 刪除舊的排程任務（如果存在）
$existingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existingTask) {
    Write-Host "移除舊的排程任務..." -ForegroundColor Yellow
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

# 建立觸發器：每天早上 7:30 執行
$Trigger = New-ScheduledTaskTrigger -Daily -At "07:30"

# 建立動作：執行 Python 腳本
$Action = New-ScheduledTaskAction `
    -Execute $PythonExe `
    -Argument "`"$PythonScript`" collect" `
    -WorkingDirectory $ScriptPath

# 建立設定：即使未登入也執行，失敗時重試
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 5)

# 取得當前使用者
$Principal = New-ScheduledTaskPrincipal `
    -UserId $env:USERNAME `
    -LogonType S4U `
    -RunLevel Highest

# 註冊排程任務
try {
    Register-ScheduledTask `
        -TaskName $TaskName `
        -Description $Description `
        -Trigger $Trigger `
        -Action $Action `
        -Settings $Settings `
        -Principal $Principal `
        -Force | Out-Null
    
    Write-Host ""
    Write-Host "✅ 排程任務設置成功！" -ForegroundColor Green
    Write-Host ""
    Write-Host "任務資訊:" -ForegroundColor Cyan
    Write-Host "  名稱: $TaskName" -ForegroundColor White
    Write-Host "  執行時間: 每天 07:30" -ForegroundColor White
    Write-Host "  執行命令: $PythonExe $PythonScript collect" -ForegroundColor White
    Write-Host ""
    Write-Host "管理排程任務:" -ForegroundColor Cyan
    Write-Host "  1. 使用工作排程器: Win + R -> taskschd.msc" -ForegroundColor White
    Write-Host "  2. 手動執行測試: Start-ScheduledTask -TaskName `"$TaskName`"" -ForegroundColor White
    Write-Host "  3. 刪除任務: Unregister-ScheduledTask -TaskName `"$TaskName`"" -ForegroundColor White
    Write-Host ""
    
    # 詢問是否立即測試
    $response = Read-Host "是否立即測試執行排程任務？(Y/N)"
    if ($response -eq "Y" -or $response -eq "y") {
        Write-Host ""
        Write-Host "執行測試..." -ForegroundColor Yellow
        Start-ScheduledTask -TaskName $TaskName
        
        Start-Sleep -Seconds 2
        
        $task = Get-ScheduledTask -TaskName $TaskName
        $taskInfo = Get-ScheduledTaskInfo -TaskName $TaskName
        
        Write-Host "任務狀態: $($task.State)" -ForegroundColor Yellow
        Write-Host "最後執行時間: $($taskInfo.LastRunTime)" -ForegroundColor Yellow
        Write-Host "最後結果: $($taskInfo.LastTaskResult)" -ForegroundColor Yellow
    }
    
} catch {
    Write-Host ""
    Write-Host "❌ 排程任務設置失敗！" -ForegroundColor Red
    Write-Host "錯誤訊息: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host ""
    Write-Host "可能的解決方案:" -ForegroundColor Yellow
    Write-Host "  1. 以系統管理員身份執行此腳本" -ForegroundColor White
    Write-Host "  2. 檢查 Python 路徑是否正確" -ForegroundColor White
    Write-Host "  3. 確保專案路徑沒有中文或特殊字元" -ForegroundColor White
    exit 1
}

Write-Host ""
Write-Host "按任意鍵結束..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
