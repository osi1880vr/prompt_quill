# Prompt Quill Ubuntu Installer

The `install_prompt_quill_ubuntu.sh` script automates the common setup steps for running Prompt Quill on an Ubuntu 22.04/24.04 workstation. It installs Python dependencies inside a virtual environment, starts the bundled Qdrant container, downloads the 19 GB snapshot from CivitAI, and uploads the data into Qdrant so the UI can run immediately.

## Prerequisites

- Ubuntu 22.04 LTS or newer
- `python3` (3.10 or 3.11) with the `venv` module (`sudo apt install python3-venv`)
- `curl`, `unzip`, `git`
- Docker Engine with the Compose plugin (`sudo apt install docker.io docker-compose-plugin`), and your user added to the `docker` group (logout/login after `sudo usermod -aG docker $USER`)
- At least 35 GB of free disk space (19 GB snapshot + cache + Python env)

## Usage

From the repository root:

```bash
chmod +x install_PQ_ubuntu/install_prompt_quill_ubuntu.sh
./install_PQ_ubuntu/install_prompt_quill_ubuntu.sh
```

The script will:

1. Check that the required commands are available.
2. Create `.venv` inside `llama_index_pq` and install the default requirements (`requirements_cpu_only.txt`).
3. Bring up `docker/qdrant/docker-compose.yaml` via `docker compose up -d`.
4. Download the Prompt Quill snapshot (prompting before the 19 GB transfer) and upload it to Qdrant.

Logs are written to `install_PQ_ubuntu/install.log`.

## Environment Variables

You can override the defaults without editing the script:

| Variable | Purpose | Default |
| --- | --- | --- |
| `PQ_REQUIREMENTS_FILE` | Relative path (inside `llama_index_pq`) to the pip requirements to install | `requirements_cpu_only.txt` |
| `PQ_VENV_DIR` | Virtual environment path | `llama_index_pq/.venv` |
| `PQ_SNAPSHOT_URL` | Download URL for the dataset zip | `https://civitai.com/api/download/models/567736` |
| `PQ_SNAPSHOT_NAME` | Snapshot filename expected inside the zip | `prompts_ng_gte-2103298935062809-2024-06-12-06-41-21.snapshot` |
| `PQ_CACHE_DIR` | Directory for downloads/extracted data | `installer_cache` |
| `PQ_QDRANT_SERVICE` | Service name from the compose file (in case you customize it) | `qdrant` |

Usage example:

```bash
PQ_REQUIREMENTS_FILE=requirements_cpu.txt ./install_PQ_ubuntu/install_prompt_quill_ubuntu.sh
```

## Starting Prompt Quill After Installation

Once the installer succeeds:

```bash
source llama_index_pq/.venv/bin/activate
python llama_index_pq/pq/prompt_quill_ui_qdrant.py
```

Make sure the Qdrant container is still running (`docker compose -f docker/qdrant/docker-compose.yaml ps`). The UI should become available at http://localhost:49152/.

## Troubleshooting

- **Docker permission denied**: ensure your user is in the `docker` group and log out/in.
- **python3-venv missing**: install with `sudo apt install python3-venv`.
- **Snapshot download fails**: rerun the script; it resumes from the cached file in `installer_cache`.
- **Upload stalls**: verify Qdrant is healthy at http://localhost:6333/health and re-run the script (it skips steps that already succeeded).

After fixing the underlying issue you can rerun the installer; it is idempotent and will skip completed steps.
