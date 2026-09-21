$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent $MyInvocation.MyCommand.Path
$visual = Join-Path $repo "visual"

Set-Location $visual

npm.cmd install
npm.cmd run setup
npm.cmd run smoke

Write-Host ""
Write-Host "Visual harness setup complete."
Write-Host "Run:"
Write-Host "  cd $visual"
Write-Host "  npm.cmd run dev"
