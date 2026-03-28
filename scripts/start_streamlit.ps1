$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$logDir = Join-Path $projectRoot "logs"
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"

function Resolve-PythonExecutable {
    param([string]$Candidate)

    if (-not $Candidate) {
        return $null
    }

    if (Test-Path $Candidate) {
        return (Resolve-Path $Candidate).Path
    }

    $command = Get-Command $Candidate -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }

    return $null
}

if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir | Out-Null
}

$pythonExe = Resolve-PythonExecutable $env:PAPER_KB_PYTHON
if (-not $pythonExe) {
    $pythonExe = Resolve-PythonExecutable $venvPython
}
if (-not $pythonExe) {
    $pythonExe = Resolve-PythonExecutable "python"
}
if (-not $pythonExe) {
    throw "Python executable not found. Set PAPER_KB_PYTHON, create .venv, or install python on PATH."
}

$arguments = @(
    "-m", "streamlit", "run", "query_interface.py",
    "--server.headless", "true",
    "--server.address", "127.0.0.1",
    "--server.port", "8501",
    "--browser.gatherUsageStats", "false"
)

$stdoutLog = Join-Path $logDir "streamlit.out.log"
$stderrLog = Join-Path $logDir "streamlit.err.log"

$process = Start-Process `
    -FilePath $pythonExe `
    -ArgumentList $arguments `
    -WorkingDirectory $projectRoot `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog `
    -PassThru

Write-Host "Started Streamlit with PID $($process.Id)"
Write-Host "Python: $pythonExe"
Write-Host "URL: http://127.0.0.1:8501"
Write-Host "Logs: $stdoutLog and $stderrLog"
