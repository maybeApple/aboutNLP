$ErrorActionPreference = "Stop"

$listenerLines = netstat -ano | Select-String "127.0.0.1:8501"
$processIds = @()

foreach ($line in $listenerLines) {
    $columns = ($line.ToString() -split "\s+") | Where-Object { $_ }
    if ($columns.Length -lt 5) {
        continue
    }
    if ($columns[0] -ne "TCP") {
        continue
    }
    if ($columns[1] -ne "127.0.0.1:8501") {
        continue
    }
    $pid = $columns[-1]
    if ($pid -match "^\d+$") {
        $processIds += [int]$pid
    }
}

$processIds = $processIds | Sort-Object -Unique

if (-not $processIds) {
    Write-Host "No Streamlit process found."
    exit 0
}

foreach ($processId in $processIds) {
    Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
    Write-Host "Stopped Streamlit PID $processId"
}
