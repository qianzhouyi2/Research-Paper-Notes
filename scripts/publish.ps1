param(
    [string]$Message = "Update wiki and rebuild Pages",
    [switch]$SkipPagesCheck
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir

Push-Location -LiteralPath $RepoRoot
try {
    python scripts/build_pages_site.py

    git restore -- '.obsidian/workspace.json' '.obsidian/plugins/agent-client/data.json' '.obsidian/plugins/agent-client/main.js' '.obsidian/plugins/agent-client/manifest.json'

    $sessionsDir = '.obsidian/plugins/agent-client/sessions'
    if (Test-Path -LiteralPath $sessionsDir) {
        Get-ChildItem -LiteralPath $sessionsDir -Filter '*.json' -File -ErrorAction SilentlyContinue |
            Remove-Item -Force -ErrorAction SilentlyContinue
    }

    git add --all -- ':/' ':(exclude).obsidian'

    git diff --cached --quiet
    if ($LASTEXITCODE -eq 0) {
        Write-Host "No staged changes; nothing to commit."
        exit 0
    }

    git commit -m $Message
    git push origin main

    if (-not $SkipPagesCheck) {
        $hasGh = Get-Command gh -ErrorAction SilentlyContinue
        if ($hasGh) {
            gh api repos/qianzhouyi2/Research-Paper-Notes/pages/builds/latest
        } else {
            Write-Host "gh not found; skipped Pages status check."
        }
    }
}
finally {
    Pop-Location
}
