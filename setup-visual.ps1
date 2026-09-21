$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent $MyInvocation.MyCommand.Path
$visual = Join-Path $repo "visual"

Set-Location $visual
npm.cmd install
npm.cmd run setup
npm.cmd run smoke

Write-Host ""
Write-Host "Starting Gym Buddy Visual Harness..."
Write-Host "Open http://127.0.0.1:5173"
npm.cmd run dev
