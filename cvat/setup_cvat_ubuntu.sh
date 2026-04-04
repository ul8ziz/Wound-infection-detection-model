#!/usr/bin/env bash
# Optional: install Docker Engine + docker-compose on Ubuntu/Debian via apt.
# Does NOT run on native Windows — use WSL2 or Docker Desktop instead.
# For CVAT clone + compose, prefer: python cvat/setup_cvat.py (from repo root).
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This script is for Linux (Ubuntu/Debian). On Windows use Docker Desktop or WSL2."
  exit 1
fi

if [[ "${EUID:-}" -ne 0 ]]; then
  echo "Run with sudo, or run: sudo bash $0"
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y docker.io docker-compose-plugin docker-compose
systemctl enable --now docker

echo "Docker installed. Add your user to the docker group if needed:"
echo "  sudo usermod -aG docker \"\$USER\" && newgrp docker"
echo "Then from the repo root: python3 cvat/setup_cvat.py"
