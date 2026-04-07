$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$javaHome = Join-Path $projectRoot "runtime\jdk-21"
$neo4jHome = Join-Path $projectRoot "runtime\neo4j"
$neo4jBat = Join-Path $neo4jHome "bin\neo4j.bat"

function Get-ComposeCommand {
    $docker = Get-Command docker -ErrorAction SilentlyContinue
    if ($docker) {
        try {
            & $docker.Source compose version *> $null
            return @($docker.Source, "compose")
        }
        catch {
        }
    }

    $dockerCompose = Get-Command docker-compose -ErrorAction SilentlyContinue
    if ($dockerCompose) {
        return @($dockerCompose.Source)
    }

    return $null
}

function Invoke-ComposeCommand {
    param(
        [string[]]$Command,
        [string[]]$Arguments
    )

    if ($Command.Length -eq 1) {
        & $Command[0] @Arguments
        return
    }

    & $Command[0] $Command[1] @Arguments
}

if ((Test-Path $neo4jBat) -and (Test-Path (Join-Path $javaHome "bin\java.exe"))) {
    $env:JAVA_HOME = $javaHome
    $env:PATH = "$javaHome\bin;$env:PATH"
    & $neo4jBat status
    exit 0
}

$composeCommand = Get-ComposeCommand
if (-not $composeCommand) {
    throw "Bundled Neo4j runtime not found, and Docker Compose is unavailable."
}

Push-Location $projectRoot
try {
    Invoke-ComposeCommand -Command $composeCommand -Arguments @("ps", "neo4j")
}
finally {
    Pop-Location
}
