# one-click installer for qualipilot on windows powershell.
#
# usage:
#     .\install.ps1              # core only
#     .\install.ps1 -Extras bedrock
#     .\install.ps1 -Extras dev
#     .\install.ps1 -Extras all
[CmdletBinding()]
param(
    [ValidateSet("core","bedrock","ollama","openai","dask","all","dev")]
    [string]$Extras = "core",
    [string]$PythonBin = "python",
    [string]$VenvDir = ".venv"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Get-Command $PythonBin -ErrorAction SilentlyContinue)) {
    throw "python not found on PATH; install Python 3.11+ first"
}

if (-not (Test-Path $VenvDir)) {
    Write-Host "creating venv at $VenvDir"
    & $PythonBin -m venv $VenvDir
}

$activate = Join-Path $VenvDir "Scripts\Activate.ps1"
. $activate

python -m pip install --upgrade pip | Out-Null

$useUv = Get-Command uv -ErrorAction SilentlyContinue
if ($useUv) {
    $installer = "uv pip install"
} else {
    Write-Host "uv not found; falling back to pip"
    $installer = "python -m pip install"
}

switch ($Extras) {
    "core" {
        Invoke-Expression "$installer -e ."
    }
    "dev" {
        Invoke-Expression "$installer -e `".[dev,all]`""
        try { pre-commit install } catch { }
    }
    default {
        Invoke-Expression "$installer -e `".[$Extras]`""
    }
}

Write-Host ""
Write-Host "installed. activate your shell with:"
Write-Host "    .\$VenvDir\Scripts\Activate.ps1"
Write-Host "then try:"
Write-Host "    qualipilot --help"
