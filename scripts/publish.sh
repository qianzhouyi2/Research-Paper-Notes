#!/usr/bin/env bash
set -euo pipefail

MESSAGE="${1:-Update wiki and rebuild Pages}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

python scripts/build_pages_site.py

git restore -- \
  '.obsidian/workspace.json' \
  '.obsidian/plugins/agent-client/data.json' \
  '.obsidian/plugins/agent-client/main.js' \
  '.obsidian/plugins/agent-client/manifest.json' || true

SESSIONS_DIR='.obsidian/plugins/agent-client/sessions'
if [[ -d "$SESSIONS_DIR" ]]; then
  shopt -s nullglob
  session_files=("$SESSIONS_DIR"/*.json)
  shopt -u nullglob
  if [[ ${#session_files[@]} -gt 0 ]]; then
    rm -f "${session_files[@]}"
  fi
fi

git add --all -- ':/' ':(exclude).obsidian'

if git diff --cached --quiet; then
  echo "No staged changes; nothing to commit."
  exit 0
fi

git commit -m "$MESSAGE"
git push origin main

if command -v gh >/dev/null 2>&1; then
  gh api repos/qianzhouyi2/Research-Paper-Notes/pages/builds/latest
else
  echo "gh not found; skipped Pages status check."
fi
