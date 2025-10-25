#!/usr/bin/env bash
set -Eeuo pipefail

# ---------- Paths and defaults ----------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
LLAMA_DIR="${REPO_ROOT}/llama_index_pq"
VENV_DIR="${PQ_VENV_DIR:-${LLAMA_DIR}/.venv}"
REQUIREMENTS_FILE="${PQ_REQUIREMENTS_FILE:-requirements_cpu_only.txt}"
CACHE_DIR="${PQ_CACHE_DIR:-${REPO_ROOT}/installer_cache}"
SNAPSHOT_URL="${PQ_SNAPSHOT_URL:-https://civitai.com/api/download/models/567736}"
SNAPSHOT_ARCHIVE="${CACHE_DIR}/data.zip"
SNAPSHOT_NAME="${PQ_SNAPSHOT_NAME:-prompts_ng_gte-2103298935062809-2024-06-12-06-41-21.snapshot}"
SNAPSHOT_PATH="${CACHE_DIR}/${SNAPSHOT_NAME}"
COMPOSE_DIR="${REPO_ROOT}/docker/qdrant"
COMPOSE_FILE="${COMPOSE_DIR}/docker-compose.yaml"
QDRANT_SERVICE="${PQ_QDRANT_SERVICE:-qdrant}"
LOG_FILE="${REPO_ROOT}/install_PQ_ubuntu/install.log"
PYTHON_BIN="${PQ_PYTHON_BIN:-python3}"

mkdir -p "$(dirname "${LOG_FILE}")" "${CACHE_DIR}"
: > "${LOG_FILE}"

log() {
	local timestamp
	timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
	echo "[${timestamp}] $*" | tee -a "${LOG_FILE}"
}

fail() {
	log "ERROR: $*"
	exit 1
}

require_path() {
	local path="$1"
	[[ -e "${path}" ]] || fail "Required path missing: ${path}"
}

require_cmd() {
	local cmd="$1"
	command -v "${cmd}" >/dev/null 2>&1 || fail "Command '${cmd}' not found. Please install it before continuing."
}

docker_compose() {
	if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
		docker compose -f "${COMPOSE_FILE}" "$@"
	elif command -v docker-compose >/dev/null 2>&1; then
		docker-compose -f "${COMPOSE_FILE}" "$@"
	else
		fail "Docker Compose is not available. Install docker compose plugin or docker-compose."
	fi
}

check_prereqs() {
	log "Checking prerequisites"
	require_cmd "${PYTHON_BIN}"
	require_cmd curl
	require_cmd unzip
	require_cmd docker
	[[ -d "${LLAMA_DIR}" ]] || fail "Cannot find ${LLAMA_DIR}. Run the script from the repository root."

	if ! groups "$USER" | grep -qw docker; then
		log "WARNING: user '${USER}' is not part of the 'docker' group. Docker commands may require sudo."
	fi
}

create_venv() {
	if [[ -d "${VENV_DIR}" ]]; then
		log "Virtual environment already exists at ${VENV_DIR}, skipping creation."
	else
		log "Creating virtual environment in ${VENV_DIR}"
		"${PYTHON_BIN}" -m venv "${VENV_DIR}" || fail "python3-venv is required. Install via 'sudo apt install python3-venv'."
	fi
}

install_requirements() {
	require_path "${LLAMA_DIR}/${REQUIREMENTS_FILE}"
	log "Upgrading pip and installing requirements from ${REQUIREMENTS_FILE}"
	"${VENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel
	"${VENV_DIR}/bin/pip" install -r "${LLAMA_DIR}/${REQUIREMENTS_FILE}"
}

start_qdrant() {
	require_path "${COMPOSE_FILE}"
	log "Starting Qdrant via docker compose"
	docker_compose up -d "${QDRANT_SERVICE}"
}

wait_for_qdrant() {
	log "Waiting for Qdrant to become ready"
	local retries=60
	while (( retries > 0 )); do
		if curl -sf "http://localhost:6333/collections" >/dev/null; then
			log "Qdrant is reachable."
			return
		fi
		sleep 3
		((retries--))
	done
	fail "Qdrant did not become ready in time."
}

confirm_large_download() {
	log "Snapshot ${SNAPSHOT_NAME} is ~19GB. Cached at ${SNAPSHOT_PATH}."
	read -r -p "Download snapshot now? (y/N): " answer
	case "${answer}" in
		[yY]|[yY][eE][sS]) return 0 ;;
		*) fail "Aborting at user request." ;;
	esac
}

download_snapshot() {
	if [[ -f "${SNAPSHOT_PATH}" ]]; then
		log "Snapshot already downloaded at ${SNAPSHOT_PATH}"
		return
	fi

	if [[ ! -f "${SNAPSHOT_ARCHIVE}" ]]; then
		confirm_large_download
		log "Downloading snapshot archive from ${SNAPSHOT_URL}"
		curl -L "${SNAPSHOT_URL}" -o "${SNAPSHOT_ARCHIVE}" || fail "Failed to download snapshot archive."
	else
		log "Using cached archive ${SNAPSHOT_ARCHIVE}"
	fi

	log "Extracting snapshot ${SNAPSHOT_NAME}"
	unzip -o "${SNAPSHOT_ARCHIVE}" "${SNAPSHOT_NAME}" -d "${CACHE_DIR}" >/dev/null || fail "Failed to extract snapshot."
	[[ -f "${SNAPSHOT_PATH}" ]] || fail "Snapshot ${SNAPSHOT_NAME} not found after extraction."
}

upload_snapshot() {
	if curl -sf "http://localhost:6333/collections/prompts_ng_gte" >/dev/null; then
		log "Collection 'prompts_ng_gte' already exists. Skipping snapshot upload."
		return
	fi

	log "Uploading snapshot to Qdrant (this may take several minutes)"
	curl -f -X POST "http://localhost:6333/collections/prompts_ng_gte/snapshots/upload" \
		-F "snapshot=@${SNAPSHOT_PATH}" \
		-o /dev/null \
		|| fail "Snapshot upload failed."
	log "Snapshot upload complete."
}

print_next_steps() {
	cat <<EOF

Installation complete!

To start Prompt Quill:
  source "${VENV_DIR}/bin/activate"
  python "${LLAMA_DIR}/pq/prompt_quill_ui_qdrant.py"

Ensure the Qdrant container stays up:
  docker compose -f "${COMPOSE_FILE}" ps

Logs: ${LOG_FILE}
EOF
}

main() {
	check_prereqs
	create_venv
	install_requirements
	start_qdrant
	wait_for_qdrant
	download_snapshot
	upload_snapshot
	print_next_steps
}

main "$@"
