# cortex-mount-watcher.ps1 — Windows variant
# Triggered by Task Scheduler when the CORTEX drive mounts.
#
# TODO: Install with Task Scheduler:
#   1. Create task triggered on Event: Microsoft-Windows-DeviceSetup, ID 112
#   2. Or use WMI event subscription for USB drive insertion
#   3. Action: powershell.exe -ExecutionPolicy Bypass -File "D:\cortex\bin\cortex-mount-watcher.ps1"
#
# Alternatively, use a scheduled task that polls for drive presence every 10s.

$ErrorActionPreference = "Stop"

# Find the CORTEX drive letter
$cortexDrive = $null
Get-PSDrive -PSProvider FileSystem | ForEach-Object {
    $testPath = Join-Path $_.Root "cortex\src"
    if (Test-Path $testPath) {
        $cortexDrive = $_.Root
    }
}

if (-not $cortexDrive) {
    Write-Host "CORTEX drive not found"
    exit 0
}

$CORTEX_HOME = Join-Path $cortexDrive "cortex"
$VENV = Join-Path $CORTEX_HOME ".venv\Scripts\python.exe"
$LOG = Join-Path $CORTEX_HOME "logs\daemon.log"
$PID_FILE = Join-Path $CORTEX_HOME "logs\daemon.pid"

# Create logs dir
New-Item -ItemType Directory -Path (Join-Path $CORTEX_HOME "logs") -Force | Out-Null

# Check if already running
if (Test-Path $PID_FILE) {
    $pid = Get-Content $PID_FILE
    try {
        $proc = Get-Process -Id $pid -ErrorAction Stop
        Write-Host "Cortex already running (PID $pid)"
        exit 0
    } catch {
        # Stale PID file, continue
    }
}

# Fall back to system Python if venv doesn't exist
if (-not (Test-Path $VENV)) {
    $VENV = "python"
}

# Run wake lifecycle
try {
    & $VENV -c @"
import sys
sys.path.insert(0, r'$CORTEX_HOME')
from src.lifecycle import wake
checkpoint = wake()
if checkpoint:
    print(f'  Resumed: boot #{checkpoint.boot_count}')
"@ 2>&1 | Out-File -Append $LOG
} catch {
    # Non-fatal
}

# Windows toast notification
try {
    [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
    $template = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02)
    $textNodes = $template.GetElementsByTagName("text")
    $textNodes.Item(0).AppendChild($template.CreateTextNode("Cortex")) | Out-Null
    $textNodes.Item(1).AppendChild($template.CreateTextNode("Daemon starting...")) | Out-Null
    $toast = [Windows.UI.Notifications.ToastNotification]::new($template)
    [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("Cortex").Show($toast)
} catch {
    # Toast not available on all Windows versions
}

# Windows TTS (SAPI)
try {
    $voice = New-Object -ComObject SAPI.SpVoice
    $voice.Speak("Cortex waking up", 1) | Out-Null  # 1 = async
} catch {
    # SAPI not available
}

# Start daemon
$process = Start-Process -FilePath $VENV -ArgumentList "-m", "src", "daemon", "--port", "11411" `
    -WorkingDirectory $CORTEX_HOME `
    -RedirectStandardOutput $LOG `
    -RedirectStandardError (Join-Path $CORTEX_HOME "logs\daemon.err") `
    -NoNewWindow -PassThru

$process.Id | Out-File $PID_FILE

# Wait for healthy (up to 15s)
for ($i = 0; $i -lt 30; $i++) {
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:11411/health" -TimeoutSec 1
        $models = $response.models_ready

        # Success toast
        try {
            $voice = New-Object -ComObject SAPI.SpVoice
            $voice.Speak("Cortex ready. $models models loaded.", 1) | Out-Null
        } catch {}

        Write-Host "Cortex ready ($models models)"
        exit 0
    } catch {
        Start-Sleep -Milliseconds 500
    }
}

Write-Host "Cortex failed to start"
try {
    $voice = New-Object -ComObject SAPI.SpVoice
    $voice.Speak("Cortex failed to start. Check logs.", 0) | Out-Null
} catch {}
exit 1
