# CVAT (self-hosted) tooling

This folder groups scripts and docs for running [CVAT](https://github.com/cvat-ai/cvat) locally with Docker.

- **`setup_cvat.py`** — Check Docker, optional install helpers, clone upstream CVAT to `../cvat`, `docker compose up`.
- **`setup_cvat_ubuntu.sh`** — Install Docker via apt on Ubuntu/Debian (`sudo` only).
- **`CVAT_SETUP.md`** — Full instructions and troubleshooting.

New annotation exports go under **`data/cvat_clean_export/`** (not under this folder); see `CVAT_SETUP.md`.

Quick start from repository root:

```bash
python cvat/setup_cvat.py --only-folders
python cvat/setup_cvat.py
```

The upstream CVAT clone defaults to a **sibling** directory `../cvat` (outside this repo), not this `cvat/` tooling folder.
