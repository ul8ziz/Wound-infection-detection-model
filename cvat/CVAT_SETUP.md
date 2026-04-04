# Self-hosted CVAT for this project

**العربية (ملخص):** البيانات الحالية في `data/original_data/` تُترك كما هي. أي تصدير أو تنظيف جديد يُوضع فقط تحت `data/cvat_clean_export/`. استخدم `cvat/setup_cvat.py` أو [دليل تثبيت CVAT](https://docs.cvat.ai/docs/administration/basics/installation/).

This repository already contains a frozen export under `data/original_data/` (CVAT tasks).  
**Any new annotation work or cleaned exports must go under `data/cvat_clean_export/`** so existing dataset files are not overwritten.

On **Ubuntu/Debian** only, you may install Docker via apt using `cvat/setup_cvat_ubuntu.sh` (run with `sudo`); then use `python cvat/setup_cvat.py` as usual. This script does nothing on Windows.

## Prerequisites

- **Docker** — [Docker Desktop](https://docs.docker.com/desktop/) on Windows or macOS; on Linux use your distribution’s Docker packages or Docker’s official repo.
- **Docker Compose V2** — included with Docker Desktop; CLI is `docker compose` (space). Legacy `docker-compose` is supported by `cvat/setup_cvat.py` if the v2 plugin is missing.
- **Git** — to clone the CVAT repository.

On **Windows**, if Docker is missing, `cvat/setup_cvat.py` will try **winget** (then Chocolatey) to install Docker Desktop. You may need to approve UAC and restart the terminal after installation. Use `--no-install-docker` to skip this and install manually.

Do **not** rely on ad-hoc `apt-get` from unrelated scripts; on Linux without root, use `sudo bash cvat/setup_cvat_ubuntu.sh` or install Docker from official docs.

## Quick path (automated folders + clone + start)

From the repository root:

```bash
python cvat/setup_cvat.py
```

Defaults:

- Clones [cvat-ai/cvat](https://github.com/cvat-ai/cvat) into a **sibling** directory `../cvat` (next to this repo).
- Creates `data/cvat_clean_export/tasks/`, `coco/`, and `splits/` for new exports only.
- Runs `docker compose up -d` inside the CVAT checkout.

**Folders only** (no Docker / no clone):

```bash
python cvat/setup_cvat.py --only-folders
```

**Custom CVAT directory:**

```bash
python cvat/setup_cvat.py --cvat-dir E:/tools/cvat
```

Other useful flags: `--skip-clone`, `--skip-up`, `--pull`, `--export-root` (override export root).

## Manual setup (official docs)

Follow the current instructions:

- [CVAT — Installation](https://docs.cvat.ai/docs/administration/basics/installation/)

After containers are healthy, open the UI (commonly **http://localhost:8080**; confirm in the docs or `docker compose` output if the port differs).

## Where to put exports

| Path | Purpose |
|------|---------|
| `data/original_data/` | **Existing** task exports — treat as read-only reference for this project’s baseline. |
| `data/cvat_clean_export/tasks/` | New per-task dumps (mirror structure you use for imports). |
| `data/cvat_clean_export/coco/` | Optional COCO JSON exports from CVAT. |
| `data/cvat_clean_export/splits/` | Optional train/val/test JSON lists for the **new** pipeline only. |

When the new data is validated, point dataset build scripts (e.g. `build_wound_focus_dataset.py`) at the new root via `--data-root` or equivalent — **after** you intentionally switch away from the legacy tree.

## Importing `wound_focus_clean` (not a “project backup”)

**Do not use “Restore project backup”** for `data/wound_focus_clean.zip` (or the `wound_focus_clean` folder). That action expects a **CVAT-native export** whose ZIP root contains **`project.json`** plus CVAT task layout. Your zip is almost certainly **images + COCO JSON** (training layout), so CVAT correctly reports: *no `project.json` in the archive*.

**Recommended flow:**

1. **Projects → Create project** — Add labels you need (see label names/colors in `data/original_data/project.json` in this repo if you want parity with the original CVAT project).
2. **Tasks → Create task** — Upload **images only** (from `data/wound_focus_clean/images/`, or a zip of those files). Finish creating the task.
3. **Open the task → Upload annotations** — Choose **COCO 1.0** (or the COCO segmentation option your CVAT version lists). Upload **`annotations_wound_only.json`** (or a split file like `train_wound_only.json` if you only imported matching images). Category IDs in the JSON must match the labels you defined in the project.
4. **Edit** — Open the job in CVAT and adjust polygons/boxes; then **export** in the format you need (e.g. COCO) into `data/cvat_clean_export/` if you are producing a cleaned revision.

**Alternative (full original CVAT tasks):** To work from the **raw** multi-task export instead, use `data/original_data/task_*` (each task has `annotations.json` in CVAT format). You can create tasks from those exports only if you use a workflow compatible with CVAT’s task/dump formats — for bulk work, importing **images + COCO** as above is usually simpler.

## Troubleshooting

- **`docker` not found** — Add Docker to PATH or restart the terminal after installing Docker Desktop. Our `cvat/setup_cvat.py` also looks under `C:\Program Files\Docker\Docker\resources\bin\`.
- **Docker Desktop: “WSL needs updating”** — Open **PowerShell or CMD as Administrator** and run `wsl --update`, then reboot if prompted. In Docker Desktop, click **Try Again**. See [Microsoft WSL documentation](https://learn.microsoft.com/windows/wsl/).
- **“WSL is not installed” / `wsl --status` fails** — As Administrator run `wsl --install`, reboot, then start Docker Desktop.
- **“Docker Desktop is unable to start”** (from `docker info`) — Same as above: install or update WSL2, reboot, open Docker Desktop until the engine is healthy, then rerun `python cvat/setup_cvat.py --skip-clone`.
- **`docker compose up` / `docker info`: cannot connect to docker API / `pipe/docker_engine`** — Start **Docker Desktop** and wait until it reports **running**. If WSL is outdated, fix with `wsl --update` first.
- **`docker-credential-desktop` not found in PATH** — Docker pulls need credential helpers next to `docker.exe`. This script prepends `C:\\Program Files\\Docker\\Docker\\resources\\bin` to `PATH` for its process; if errors persist, add that folder to **system** environment PATH and restart the terminal.
- **`docker compose` not found** — Enable Compose V2 in Docker Desktop settings or install the plugin per Docker documentation.
- **Port already in use** — Stop other services on the same port or adjust CVAT’s compose configuration per upstream docs.

### Browser: “Cannot connect to the server” (CVAT UI)

This means the **frontend cannot reach the API** — often the stack is still starting, a container crashed, or the URL/port is wrong.

1. **Confirm containers are up** (in your CVAT clone directory, e.g. `E:\GitHub\cvat` — sibling of this repo by default):
   ```bash
   docker compose ps
   ```
   All core services should be `running` or `healthy` (first start after `up -d` can take **several minutes** while Postgres, Redis, ClickHouse, etc. initialize).

2. **Inspect logs** if any service exits or stays unhealthy (service names depend on CVAT version):
   ```bash
   docker compose logs --tail=120
   ```
   Repeat with `-f` on a specific service if `docker compose ps` shows one container restarting.

3. **Use the correct URL** — Usually **http://localhost:8080** (or the port shown in CVAT’s `docker compose` / docs). Avoid `https` unless you configured TLS.

4. **Restart the stack** after fixing Docker/WSL:
   ```bash
   docker compose down
   docker compose up -d
   ```

5. **Upgrades** — If you upgraded CVAT from an old release, follow the official [Upgrade Guide](https://docs.cvat.ai/docs/administration/advanced/upgrade/) (linked from the in-app message for migrations from 2.2.0 or earlier).

6. **Run the helper from the repo root** (not the old `scripts/` path):
   ```bash
   python cvat/setup_cvat.py --skip-clone
   ```
   Ensure `docker info` works before composing.
