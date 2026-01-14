# PowerShell helper: create venv and install requirements

param(
    [switch]$ForceExecutionPolicy
)

$venvPath = ".venv"
if (-Not (Test-Path $venvPath)) {
    python -m venv $venvPath
}

if ($ForceExecutionPolicy) {
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
}

# Activate the venv for the current PowerShell session
.\$venvPath\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt

Write-Host "Environment ready. To activate later run: .\$venvPath\Scripts\Activate.ps1" -ForegroundColor Green
