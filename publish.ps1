param(
    [string]$Message = "update site"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

$status = git status --short -- docs
if (-not $status) {
    Write-Host "No changes under docs to publish."
    exit 0
}

git add docs
git commit -m $Message
git push
