# Run every weekday at 8:00 AM IST (after Asian open scan, before US open)
# Imports this script into Windows Task Scheduler.
# Run from PowerShell (one-time setup):
#   powershell -ExecutionPolicy Bypass -File .\schedule_task.ps1

$ProjectRoot = $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Script = Join-Path $ProjectRoot "daily_runner.py"
$LogFile = Join-Path $ProjectRoot "daily.log"

if (-not (Test-Path $Python)) {
    Write-Error "Create venv first: python -m venv .venv ; .\.venv\Scripts\pip install -r requirements.txt"
    exit 1
}

$action = New-ScheduledTaskAction -Execute $Python `
    -Argument "`"$Script`"" `
    -WorkingDirectory $ProjectRoot

$trigger = New-ScheduledTaskTrigger -Daily -At 8:00am
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -RunOnlyIfNetworkAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 2)

Register-ScheduledTask -TaskName "DailyStockAnalysis" `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Description "Daily Indian + US stock analysis with Telegram report" `
    -Force

Write-Host "✅ Scheduled task 'DailyStockAnalysis' created. Runs daily at 8:00 AM."
Write-Host "   Manage at: Task Scheduler → Task Scheduler Library → DailyStockAnalysis"
