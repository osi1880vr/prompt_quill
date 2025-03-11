import os
import shutil
import subprocess
import sys
import zipfile
import logging
from typing import Optional
from pathlib import Path
from tqdm import tqdm
import time

# Force plain text logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# Lazy import with automatic installation
for module, package in [("requests", "requests"), ("tqdm", "tqdm")]:
    try:
        __import__(module)
    except ModuleNotFoundError:
        logger.warning(f"{module} library not found. Installing...")
        # Use pip from the current environment
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--no-cache-dir"])
import requests
from tqdm import tqdm

# Setup directories
BASE_DIR = Path.cwd()
INSTALL_DIR = BASE_DIR / "installer_files"
CACHE_DIR = BASE_DIR / "installer_cache"
for directory in (INSTALL_DIR, CACHE_DIR):
    directory.mkdir(exist_ok=True)

def get_available_space(path: Path) -> int:
    """Get available disk space in bytes."""
    try:
        stat = os.statvfs(str(path)) if os.name == 'posix' else shutil.disk_usage(str(path))
        return stat.f_bavail * stat.f_frsize if os.name == 'posix' else stat.free
    except Exception as e:
        logger.error(f"Failed to check disk space: {str(e)}")
        return float('inf')

def download_file(url: str, output_path: Path, max_retries: int = 5, timeout: int = 30) -> None:
    """Download a file with retries, progress bar, and size verification."""
    output_path.parent.mkdir(exist_ok=True)
    logger.info(f"Attempt download {url}")
    try:
        response = requests.head(url, timeout=timeout)
        total_size = int(response.headers.get('content-length', 0))
        if total_size:
            available_space = get_available_space(output_path.parent)
            if available_space < total_size * 1.1:
                logger.error(f"Insufficient disk space: {available_space / (1024**3):.2f}GB available, {total_size / (1024**3):.2f}GB needed")
                sys.exit(1)
    except requests.RequestException as e:
        logger.warning(f"Could not check file size: {str(e)}. Proceeding with download.")

    attempt = 0
    while attempt < max_retries:
        try:
            attempt += 1
            logger.info(f"Attempt {attempt} to download {url}")
            with requests.get(url, stream=True, timeout=timeout, verify=True) as response:
                response.raise_for_status()
                total_size = int(response.headers.get('content-length', 0))

                # Plain text tqdm - no ANSI
                with open(output_path, 'wb') as f, tqdm(
                        desc=output_path.name,
                        total=total_size,
                        unit='iB',
                        unit_scale=True,
                        unit_divisor=1024,
                        ascii=True,
                        file=sys.stdout,
                        colour=None
                ) as bar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            size = f.write(chunk)
                            bar.update(size)

                downloaded_size = output_path.stat().st_size
                if total_size and downloaded_size != total_size:
                    logger.error(f"Download incomplete: {downloaded_size} vs {total_size} bytes")
                    sys.exit(1)
                logger.info(f"Download completed: {output_path}")
                return

        except (requests.RequestException, ValueError, OSError) as e:
            logger.error(f"Download failed: {str(e)}")
            if attempt == max_retries:
                logger.critical(f"Max retries ({max_retries}) exceeded for {url}")
                if output_path.exists():
                    output_path.unlink()
                sys.exit(1)
            time.sleep(2 ** attempt)
    sys.exit(1)

def extract_zip(zip_path: Path, extract_to: Path) -> None:
    """Extract a zip file with error handling."""
    try:
        extract_to.mkdir(exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        logger.info(f"Extracted {zip_path} to {extract_to}")
    except (zipfile.BadZipFile, OSError) as e:
        logger.error(f"Failed to extract {zip_path}: {str(e)}")
        sys.exit(1)

def download_qdrant() -> None:
    """Download and extract Qdrant components."""
    qdrant_dir = INSTALL_DIR / 'qdrant'
    snapshot_name = 'prompts_ng_gte-2103298935062809-2024-06-12-06-41-21.snapshot'
    loaded_flag = INSTALL_DIR / 'qdrant_loaded.txt'
    temp_dir = INSTALL_DIR / 'temp_extract'

    progress = 0

    def update_progress(stage_name: str, increment: int):
        nonlocal progress
        progress += increment
        if progress > 100:
            progress = 100
        logger.info(f"Qdrant Install Progress: {progress}%% - {stage_name}")
        print(f"Qdrant Install Progress: {progress}%")

    downloads = [
        {
            'name': 'Qdrant',
            'url': 'https://github.com/qdrant/qdrant/releases/download/v1.12.6/qdrant-x86_64-pc-windows-msvc.zip',
            'filename': 'qdrant-x86_64-pc-windows-msvc.zip',
            'expected_file': qdrant_dir / 'qdrant.exe'
        },
        {
            'name': 'Qdrant Web UI',
            'url': 'https://github.com/qdrant/qdrant-web-ui/releases/download/v0.1.33/dist-qdrant.zip',
            'filename': 'dist-qdrant.zip',
            'expected_file': qdrant_dir / 'static' / 'index.html'
        },
        {
            'name': 'Prompt Quill data',
            'url': 'https://civitai.com/api/download/models/567736',
            'filename': 'data.zip',
            'expected_file': temp_dir / snapshot_name
        }
    ]

    for item in downloads:
        cache_path = CACHE_DIR / item['filename']
        install_path = INSTALL_DIR / item['filename']
        is_big_file = item['name'] == 'Prompt Quill data'

        if is_big_file:
            logger.info(f"Starting download of {item['name']} - this is a 19GB file and may take a while, please be patient!")
        else:
            logger.info(f"Starting download of {item['name']} - please wait...")

        if not cache_path.exists() and not loaded_flag.exists():
            logger.info(f"Downloading {item['name']} from {item['url']}")
            download_file(item['url'], install_path)
            if not install_path.exists():
                logger.error(f"{item['name']} download failed - file missing")
                sys.exit(1)
            shutil.copy(install_path, cache_path)
        else:
            logger.info(f"Using cached {item['name']}")
            if not install_path.exists():
                shutil.copy(cache_path, install_path)
        update_progress(f"Downloaded {item['name']}", 25)
        logger.info(f"Finished downloading {item['name']} - success!")

    qdrant_exe_zip = INSTALL_DIR / 'qdrant-x86_64-pc-windows-msvc.zip'
    qdrant_dist_zip = INSTALL_DIR / 'dist-qdrant.zip'
    data_zip = INSTALL_DIR / 'data.zip'

    logger.info("Starting extraction of Qdrant executable - please wait...")
    if not (qdrant_dir / 'qdrant.exe').exists():
        if not qdrant_exe_zip.exists():
            logger.error("Qdrant zip file missing")
            sys.exit(1)
        extract_zip(qdrant_exe_zip, qdrant_dir)
        if not (qdrant_dir / 'qdrant.exe').exists():
            logger.error("Qdrant.exe not found after extraction")
            sys.exit(1)
    logger.info("Qdrant executable extracted successfully!")

    logger.info("Starting extraction of Qdrant Web UI - please wait...")
    if not (qdrant_dir / 'static').exists():
        if not qdrant_dist_zip.exists():
            logger.error("Qdrant Web UI zip file missing")
            sys.exit(1)
        extract_zip(qdrant_dist_zip, qdrant_dir)
        if (qdrant_dir / 'dist').exists():
            (qdrant_dir / 'dist').rename(qdrant_dir / 'static')
        if not (qdrant_dir / 'static').exists():
            logger.error("Static dir not found after extraction")
            sys.exit(1)
    logger.info("Qdrant Web UI extracted successfully!")

    snapshot_path = temp_dir / snapshot_name
    if not snapshot_path.exists() and not loaded_flag.exists():
        logger.info("Starting extraction of Prompt Quill data - this is a 19GB file and may take several minutes, please be patient!")
        if not data_zip.exists():
            logger.error("Data zip file missing")
            sys.exit(1)
        extract_zip(data_zip, temp_dir)
        if not snapshot_path.exists():
            logger.error(f"Snapshot {snapshot_name} not found after extraction")
            sys.exit(1)
        logger.info("Prompt Quill data extracted successfully - great job waiting that out!")
    else:
        logger.info("Prompt Quill data snapshot already exists - skipping extraction.")

    update_progress("Extraction complete", 25)
    logger.info("Downloads and extraction complete - Qdrant setup will proceed in batch script")

if __name__ == "__main__":
    try:
        download_qdrant()
    except KeyboardInterrupt:
        logger.warning("Installation interrupted by user")
        sys.exit(1)