#!/usr/bin/env bash
set -euo pipefail

if ! command -v ollama >/dev/null 2>&1; then
  curl -fsSL https://ollama.com/install.sh | sh
fi

ollama pull clef:latest
ollama pull clef-flash:latest

echo "Ollama is ready with clef:latest and clef-flash:latest."
