"""
Optional helper: verify Docker, clone CVAT, start containers, create export folders.

When Docker is missing, the script tries to install it (Windows: winget/Chocolatey;
Linux: apt if run as root; macOS: prints manual link).

New annotation exports go under data/cvat_clean_export/ — existing data/original_data/ is untouched.

Usage (from repository root):
    python cvat/setup_cvat.py --only-folders
    python cvat/setup_cvat.py
    python cvat/setup_cvat.py --no-install-docker
    python cvat/setup_cvat.py --cvat-dir ../cvat --skip-up

See cvat/CVAT_SETUP.md for full instructions.
"""

from __future__ import annotations

import argparse
import logging
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_CVAT_CLONE = "https://github.com/cvat-ai/cvat.git"
DEFAULT_EXPORT_REL = Path("data/cvat_clean_export")
WIN_DOCKER_WINGET_ID = "Docker.DockerDesktop"
DOCKER_DESKTOP_WINDOWS_URL = "https://docs.docker.com/desktop/install/windows-install/"

# Resolved path to docker.exe (Windows) or "docker" when found on PATH.
_DOCKER_EXE: str = "docker"


def docker_exe() -> str:
    return _DOCKER_EXE


def prepend_docker_cli_dir_to_path() -> None:
    """
    Put Docker's bin directory on PATH so helpers like docker-credential-desktop.exe
    are found when pulling images (IDE terminals sometimes omit it).
    """
    bin_dir = Path(docker_exe()).resolve().parent
    if not bin_dir.is_dir():
        return
    key = "PATH"
    current = os.environ.get(key, "")
    prefix = str(bin_dir) + os.pathsep
    if current.startswith(prefix):
        return
    os.environ[key] = prefix + current
    logging.debug("Prepended Docker CLI directory to PATH: %s", bin_dir)


def run_cmd(
    cmd: List[str],
    cwd: Optional[Path] = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    logging.debug("Running: %s (cwd=%s)", " ".join(cmd), cwd)
    return subprocess.run(
        cmd,
        cwd=cwd,
        check=check,
        capture_output=True,
        text=True,
    )


def discover_docker_cli() -> bool:
    """
    Find a working Docker CLI: PATH first, then typical Docker Desktop paths on Windows.
    Updates module _DOCKER_EXE on success.
    """
    global _DOCKER_EXE
    candidates: List[str] = []
    w = shutil.which("docker")
    if w:
        candidates.append(w)
    if platform.system() == "Windows":
        pf = os.environ.get("ProgramFiles", r"C:\Program Files")
        candidates.append(
            str(Path(pf) / "Docker" / "Docker" / "resources" / "bin" / "docker.exe")
        )
    seen: set[str] = set()
    for c in candidates:
        if not c or c in seen:
            continue
        seen.add(c)
        path = Path(c)
        if not path.is_file():
            continue
        try:
            run_cmd([str(path), "--version"])
            _DOCKER_EXE = str(path)
            logging.info("Using Docker CLI: %s", _DOCKER_EXE)
            return True
        except FileNotFoundError:
            continue
        except subprocess.CalledProcessError:
            continue
    _DOCKER_EXE = "docker"
    logging.warning("Docker CLI not found in PATH or default install locations.")
    return False


def _install_docker_windows() -> bool:
    """Try winget, then Chocolatey. May require elevation; user may need to restart terminal."""
    winget = shutil.which("winget")
    if winget:
        logging.info("Installing Docker Desktop via winget (%s)...", WIN_DOCKER_WINGET_ID)
        try:
            r = subprocess.run(
                [
                    winget,
                    "install",
                    "-e",
                    "--id",
                    WIN_DOCKER_WINGET_ID,
                    "--accept-package-agreements",
                    "--accept-source-agreements",
                ],
                check=False,
            )
            if r.returncode == 0:
                logging.warning(
                    "Docker Desktop installed. Start it from the Start menu, wait until it "
                    "finishes starting, then open a new terminal and run this script again."
                )
            else:
                logging.warning(
                    "winget exited with code %s (often already installed); checking for CLI...",
                    r.returncode,
                )
        except OSError as e:
            logging.warning("winget failed: %s", e)

    choco = shutil.which("choco")
    if choco:
        logging.info("Installing Docker Desktop via Chocolatey...")
        try:
            r = subprocess.run(
                [choco, "install", "docker-desktop", "-y"],
                check=False,
            )
            if r.returncode == 0:
                logging.warning(
                    "Docker Desktop installed. Start Docker Desktop, then open a new "
                    "terminal and run this script again."
                )
        except OSError as e:
            logging.warning("choco failed: %s", e)

    if discover_docker_cli():
        return True

    logging.error(
        "Could not install Docker automatically. Install manually: %s",
        DOCKER_DESKTOP_WINDOWS_URL,
    )
    return False


def _install_docker_linux() -> bool:
    """apt install when running as root; otherwise instruct user."""
    if os.geteuid() != 0:
        ubuntu_script = SCRIPT_DIR / "setup_cvat_ubuntu.sh"
        logging.error(
            "Docker not installed. On Ubuntu/Debian run:\n"
            "  sudo bash %s\n"
            "or: sudo apt install -y docker.io docker-compose-plugin",
            ubuntu_script,
        )
        return False

    logging.info("Installing Docker via apt (running as root)...")
    try:
        subprocess.run(
            ["apt-get", "update", "-y"],
            check=True,
        )
        subprocess.run(
            [
                "apt-get",
                "install",
                "-y",
                "docker.io",
                "docker-compose-plugin",
            ],
            check=True,
        )
        subprocess.run(["systemctl", "enable", "--now", "docker"], check=True)
        logging.info("Docker installed. Add your user to group docker: sudo usermod -aG docker $USER")
        return True
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        logging.error("apt install failed: %s", e)
        return False


def _install_docker_darwin() -> bool:
    logging.error(
        "Install Docker Desktop for Mac: https://docs.docker.com/desktop/install/mac-install/"
    )
    return False


def try_install_docker() -> bool:
    """Platform-specific install attempt. Returns True if install command succeeded (CLI may still be missing until reboot)."""
    system = platform.system()
    if system == "Windows":
        return _install_docker_windows()
    if system == "Linux":
        return _install_docker_linux()
    if system == "Darwin":
        return _install_docker_darwin()
    logging.error("Unknown OS; install Docker manually: https://docs.docker.com/get-docker/")
    return False


def ensure_docker(*, allow_install: bool) -> bool:
    """Return True if docker CLI is available, optionally triggering install first."""
    if discover_docker_cli():
        return True
    if not allow_install:
        if platform.system() == "Windows":
            logging.error(
                "Docker is not installed or not on PATH. Install manually or re-run "
                "without --no-install-docker: %s",
                DOCKER_DESKTOP_WINDOWS_URL,
            )
        else:
            logging.error(
                "Docker is not installed or not on PATH. Install manually or re-run "
                "without --no-install-docker: https://docs.docker.com/get-docker/"
            )
        return False
    if not try_install_docker():
        if discover_docker_cli():
            return True
        return False
    if discover_docker_cli():
        return True
    logging.error(
        "Docker was installed or the installer finished, but the Docker CLI is still missing. "
        "Start Docker Desktop from the Start menu, wait until it is running, then run this "
        "script again (a new terminal is not always required if docker.exe is found)."
    )
    return False


def docker_compose_invocation() -> Optional[List[str]]:
    """
    Prefer Docker Compose V2 (`docker compose`).
    Fall back to standalone `docker-compose` if present.
    """
    d = docker_exe()
    try:
        run_cmd([d, "compose", "version"])
        return [d, "compose"]
    except FileNotFoundError:
        pass
    except subprocess.CalledProcessError:
        pass

    try:
        run_cmd(["docker-compose", "--version"])
        return ["docker-compose"]
    except FileNotFoundError:
        logging.error("Neither `docker compose` nor `docker-compose` is available.")
        return None
    except subprocess.CalledProcessError as e:
        logging.error("docker-compose check failed: %s", e.stderr or e.stdout or e)
        return None


def check_docker_daemon() -> bool:
    """
    Verify the Docker engine is reachable (Docker Desktop finished starting).
    Avoids long compose pulls when the daemon is down (WSL/outdated Desktop).
    """
    try:
        r = subprocess.run(
            [docker_exe(), "info"],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if r.returncode == 0:
            return True
        combined = ((r.stderr or "") + (r.stdout or "")).lower()
        logging.error("Docker engine is not reachable (docker info failed).")
        if "unable to start" in combined:
            logging.error(
                "Docker Desktop could not start. On Windows: fix WSL2 - open PowerShell "
                "as Administrator and run:  wsl --install   OR   wsl --update"
            )
            logging.error("Reboot if prompted, then start Docker Desktop and run this script again.")
        elif "dockerdesktoplinuxengine" in combined or "docker_engine" in combined:
            logging.error(
                "Docker Linux backend is not running. If WSL is missing: PowerShell as "
                "Administrator:  wsl --install  then reboot. If WSL exists:  wsl --update"
            )
            logging.error(
                "Then start Docker Desktop until the whale icon is steady, and run this script again."
            )
        elif "//./pipe/" in combined:
            logging.error(
                "Start Docker Desktop from the Start menu; wait until it is running, then: "
                "python cvat/setup_cvat.py --skip-clone"
            )
        elif "wsl" in combined and "not installed" in combined:
            logging.error(
                "WSL is not installed. As Administrator:  wsl --install  then reboot."
            )
        else:
            tail = combined[:1200] if combined else "(no output)"
            logging.error("Details: %s", tail)
        return False
    except subprocess.TimeoutExpired:
        logging.error("docker info timed out — Docker Desktop may be stuck starting.")
        return False
    except FileNotFoundError:
        return False


def default_cvat_dir() -> Path:
    """Sibling folder next to the project repo: <parent>/cvat."""
    return PROJECT_ROOT.parent / "cvat"


def download_cvat(cvat_dir: Path, clone_url: str, pull_if_exists: bool) -> None:
    if cvat_dir.exists() and any(cvat_dir.iterdir()):
        logging.info("CVAT directory already exists: %s", cvat_dir)
        if pull_if_exists:
            logging.info("Running git pull...")
            try:
                run_cmd(["git", "pull"], cwd=cvat_dir)
            except FileNotFoundError:
                logging.warning("git not in PATH; skipping pull.")
            except subprocess.CalledProcessError as e:
                logging.warning("git pull failed: %s", e.stderr or e.stdout or e)
        return

    cvat_dir.parent.mkdir(parents=True, exist_ok=True)
    logging.info("Cloning CVAT from %s into %s", clone_url, cvat_dir)
    try:
        run_cmd(["git", "clone", clone_url, str(cvat_dir)])
    except FileNotFoundError:
        logging.error("git not found in PATH. Install Git or clone CVAT manually.")
        raise SystemExit(1) from None
    except subprocess.CalledProcessError as e:
        logging.error("git clone failed: %s", e.stderr or e.stdout or e)
        raise SystemExit(1) from e


def run_cvat(cvat_dir: Path, compose_cmd: List[str]) -> None:
    compose_file = cvat_dir / "docker-compose.yml"
    if not compose_file.is_file():
        logging.error(
            "Expected %s — clone CVAT first or use --cvat-dir pointing to a CVAT checkout.",
            compose_file,
        )
        raise SystemExit(1)
    logging.info("Starting CVAT (%s up -d)...", compose_cmd[0])
    try:
        subprocess.run(
            compose_cmd + ["up", "-d"],
            cwd=cvat_dir,
            check=True,
        )
    except FileNotFoundError:
        logging.error("Compose command not found: %s", compose_cmd[0])
        raise SystemExit(1) from None
    except subprocess.CalledProcessError as e:
        logging.error("docker compose up failed with exit code %s", e.returncode)
        logging.error(
            "If you see 'docker API' or 'pipe/docker_engine', start Docker Desktop and wait "
            "until it is running, then run: python cvat/setup_cvat.py --skip-clone"
        )
        logging.error(
            "If you see 'docker-credential-desktop' not in PATH, this script prepends "
            "Docker's bin folder; restart the terminal or add it to your system PATH."
        )
        raise SystemExit(1) from e


def create_export_folders(export_root: Path) -> None:
    """
    Create data/cvat_clean_export/{tasks,coco,splits} under the project.
    Does not write to data/original_data/ or other existing dataset roots.
    """
    sub = ["tasks", "coco", "splits"]
    export_root.mkdir(parents=True, exist_ok=True)
    for name in sub:
        (export_root / name).mkdir(parents=True, exist_ok=True)
    logging.info("Created export tree under %s", export_root.resolve())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CVAT Docker helper and export folder setup.")
    p.add_argument(
        "--cvat-dir",
        type=Path,
        default=default_cvat_dir(),
        help="CVAT clone directory (default: sibling ../cvat next to project root).",
    )
    p.add_argument(
        "--clone-url",
        default=DEFAULT_CVAT_CLONE,
        help="Git URL for CVAT (default: upstream cvat-ai/cvat).",
    )
    p.add_argument(
        "--export-root",
        type=Path,
        default=PROJECT_ROOT / DEFAULT_EXPORT_REL,
        help="Root for new exports/cleaned data (default: <repo>/data/cvat_clean_export).",
    )
    p.add_argument(
        "--only-folders",
        action="store_true",
        help="Only create data/cvat_clean_export/*; skip Docker and clone.",
    )
    p.add_argument("--skip-clone", action="store_true", help="Do not clone or pull CVAT.")
    p.add_argument("--skip-up", action="store_true", help="Do not run docker compose up -d.")
    p.add_argument(
        "--pull",
        action="store_true",
        help="If CVAT dir exists, run git pull before compose.",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Debug logging.",
    )
    p.add_argument(
        "--no-install-docker",
        action="store_true",
        help="Do not try to install Docker when the CLI is missing (winget/choco/apt).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    create_export_folders(args.export_root.resolve())

    if args.only_folders:
        logging.info("Done (--only-folders).")
        return

    allow_install = not args.no_install_docker
    if not ensure_docker(allow_install=allow_install):
        raise SystemExit(1)

    prepend_docker_cli_dir_to_path()

    compose = docker_compose_invocation()
    if compose is None:
        logging.error(
            "Install Docker Compose V2 plugin (`docker compose`) or docker-compose."
        )
        raise SystemExit(1)

    if not check_docker_daemon():
        raise SystemExit(1)

    if not args.skip_clone:
        download_cvat(args.cvat_dir.resolve(), args.clone_url, pull_if_exists=args.pull)

    if not args.skip_up:
        run_cvat(args.cvat_dir.resolve(), compose)

    logging.info(
        "CVAT UI is usually at http://localhost:8080 (see CVAT docs if your port differs)."
    )
    logging.info(
        "Export new annotations under %s — do not overwrite data/original_data/.",
        args.export_root.resolve(),
    )


if __name__ == "__main__":
    main()
