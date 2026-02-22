$TargetScript = Join-Path $PSScriptRoot "每日自動執行.bat"
$WorkDir = $PSScriptRoot
$TaskName = "AI_Stock_Auto_Trader"

Write-Host "正在設定 Windows 排程以自動化執行..."

# 定義動作：執行 bat 檔
$action = New-ScheduledTaskAction -Execute $TargetScript -WorkingDirectory $WorkDir

# 定義觸發條件：每週一至週五 下午 14:30 (收盤後)
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At 2:30PM

# 設定：允許使用電池時執行，若錯過時間(如關機)則開機後儘快執行
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 1)

# 註冊任務 (若已存在則覆蓋)
try {
    Register-ScheduledTask -Action $action -Trigger $trigger -Settings $settings -TaskName $TaskName -Description "AI 股票機器人每日自動掃描與交易" -Force
    Write-Host "✅ 排程設定成功！"
    Write-Host "您的 AI 機器人現在將於每週一至週五的下午 2:30 自動啟動。"
    Write-Host "您可以在 Windows 的「工作排程器」中找到 '$TaskName' 進行管理。"
} catch {
    Write-Host "❌ 設定失敗。請嘗試以「系統管理員身分」執行此腳本。"
    Write-Host "錯誤訊息: $_"
}

Pause
