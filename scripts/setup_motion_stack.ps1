$ErrorActionPreference = "Stop"

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

& wsl.exe --status | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "WSL2 is required. Install it first with: wsl --install -d Ubuntu"
}

$wslRepo = (& wsl.exe wslpath -a -u $repo).Trim()
if (-not $wslRepo) {
    throw "Could not translate repo path into WSL path."
}

$escaped = $wslRepo.Replace("'", "'\''")
& wsl.exe bash -lc "cd '$escaped' && bash scripts/setup_motion_stack_wsl.sh"
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
