# Install qualipilot from this checkout on Windows.
#
# usage:
#     .\install.ps1
#     .\install.ps1 -Extras bedrock,linking,duckdb
#     .\install.ps1 -Extras all
#     .\install.ps1 -Dev
[CmdletBinding()]
param(
    [ValidateSet(
        "bedrock",
        "dask",
        "ollama",
        "openai",
        "linking",
        "duckdb",
        "spark",
        "all"
    )]
    [string[]]$Extras = @(),
    [switch]$Dev,
    [string]$PythonBin = "python",
    [string]$VenvDir = ".venv"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Invoke-Checked {
    param(
        [Parameter(Mandatory)]
        [string]$Command,
        [Parameter(Mandatory)]
        [string[]]$Arguments
    )

    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Command failed with exit code $LASTEXITCODE"
    }
}

if (-not (Get-Command $PythonBin -ErrorAction SilentlyContinue)) {
    throw "$PythonBin not found; install Python 3.11-3.13 first"
}

Invoke-Checked -Command $PythonBin -Arguments @(
    "-c",
    "import sys; raise SystemExit(not ((3, 11) <= sys.version_info < (3, 14)))"
)

if (-not (Test-Path -LiteralPath $VenvDir -PathType Container)) {
    Write-Host "creating virtual environment at $VenvDir"
    Invoke-Checked -Command $PythonBin -Arguments @("-m", "venv", $VenvDir)
}

$activate = Join-Path $VenvDir "Scripts\Activate.ps1"
. $activate
Invoke-Checked -Command "python" -Arguments @(
    "-c",
    "import sys; raise SystemExit(not ((3, 11) <= sys.version_info < (3, 14)))"
)

$selectedExtras = [System.Collections.Generic.List[string]]::new()
foreach ($extra in $Extras) {
    $selectedExtras.Add($extra)
}

if (-not (Get-Command "uv" -ErrorAction SilentlyContinue)) {
    Write-Host "uv not found; installing uv 0.11.21"
    Invoke-Checked -Command "python" -Arguments @(
        "-m",
        "pip",
        "install",
        "uv==0.11.21"
    )
}

$syncArgs = @("sync", "--locked", "--active")
if (-not $Dev) {
    $syncArgs += @("--no-dev", "--no-editable")
}
foreach ($extra in $selectedExtras) {
    $syncArgs += @("--extra", $extra)
}
Invoke-Checked -Command "uv" -Arguments $syncArgs

if ($Dev) {
    Invoke-Checked -Command "pre-commit" -Arguments @("install")
}

Write-Host ""
Write-Host "installed. activate the environment with:"
Write-Host "    . $activate"
Write-Host "then run:"
Write-Host "    qualipilot --help"
