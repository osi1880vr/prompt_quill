import os
import shutil
import subprocess
import sys
import time
import zipfile
import logging
from typing import Optional
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('installation.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Lazy import with automatic installation
for module, package in [("requests", "requests"), ("tqdm", "tqdm")]:
    if not __import__(module, fromlist=['']).__name__:
        logger.warning(f"{module} library not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
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

def download_file(url: str, output_path: Path, max_retries: int = 5, timeout: int = 30) -> Optional[int]:
    """Download a file with retries, progress bar, and size verification."""
    output_path.parent.mkdir(exist_ok=True)

    # Disk space check
    try:
        response = requests.head(url, timeout=timeout)
        total_size = int(response.headers.get('content-length', 0))
        if total_size:
            available_space = get_available_space(output_path.parent)
            if available_space < total_size * 1.1:  # 10% buffer
                logger.error(f"Insufficient disk space: {available_space / (1024**3):.2f}GB available, {total_size / (1024**3):.2f}GB needed")
                raise RuntimeError("Not enough disk space")
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

                with open(output_path, 'wb') as f, tqdm(
                        desc=output_path.name,
                        total=total_size,
                        unit='iB',
                        unit_scale=True,
                        unit_divisor=1024,
                ) as bar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            size = f.write(chunk)
                            bar.update(size)

                downloaded_size = output_path.stat().st_size
                if total_size and downloaded_size != total_size:
                    raise ValueError(f"Download incomplete: {downloaded_size} vs {total_size} bytes")

                logger.info(f"Download completed: {output_path}")
                return response.status_code

        except (requests.RequestException, ValueError, OSError) as e:
            logger.error(f"Download failed: {str(e)}")
            if attempt == max_retries:
                retry = input("Max retries exceeded. Retry download? (y/n): ").lower().strip() == 'y'
                if retry:
                    attempt = 0
                    continue
                logger.critical(f"Max retries ({max_retries}) exceeded for {url}")
                if output_path.exists():
                    output_path.unlink()
                return None
            time.sleep(2 ** attempt)  # Exponential backoff
    return None  # This line is technically unreachable due to the while loop, but kept for clarity

def extract_zip(zip_path: Path, extract_to: Path) -> None:
    """Extract a zip file with error handling."""
    try:
        extract_to.mkdir(exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        logger.info(f"Extracted {zip_path} to {extract_to}")
    except (zipfile.BadZipFile, OSError) as e:
        logger.error(f"Failed to extract {zip_path}: {str(e)}")
        raise

def download_qdrant() -> None:
    """Download and setup Qdrant components."""
    qdrant_dir = INSTALL_DIR / 'qdrant'
    snapshot_name = 'prompts_ng_gte-2103298935062809-2024-06-12-06-41-21.snapshot'
    loaded_flag = INSTALL_DIR / 'qdrant_loaded.txt'

    # Progress tracking
    stages = 4  # Download Qdrant, Web UI, Data, Extraction
    stage_weight = 25
    progress = 0

    def update_progress(stage_name: str):
        nonlocal progress
        progress += stage_weight
        logger.info(f"Qdrant Install Progress: {progress}% - {stage_name}")
        print(f"Qdrant Install Progress: {progress}%")

    downloads = [
        {
            'name': 'Qdrant',
            'url': 'https://github.com/qdrant/qdrant/releases/download/v1.12.6/qdrant-x86_64-pc-windows-msvc.zip',
            'filename': 'qdrant-x86_64-pc-windows-msvc.zip'
        },
        {
            'name': 'Qdrant Web UI',
            'url': 'https://github.com/qdrant/qdrant-web-ui/releases/download/v0.1.33/dist-qdrant.zip',
            'filename': 'dist-qdrant.zip'
        },
        {
            'name': 'Prompt Quill data',
            'url': 'https://civitai.com/api/download/models/567736',
            'filename': 'data.zip'
        }
    ]

    try:
        for item in downloads:
            cache_path = CACHE_DIR / item['filename']
            install_path = INSTALL_DIR / item['filename']

            if not cache_path.exists() and not loaded_flag.exists():
                logger.info(f"Downloading {item['name']}")
                status = download_file(item['url'], install_path)
                if status != 200:
                    raise RuntimeError(f"Failed to download {item['name']} - HTTP {status}")
                shutil.copy(install_path, cache_path)
            else:
                logger.info(f"Using cached {item['name']}")
                if not install_path.exists():
                    shutil.copy(cache_path, install_path)
            update_progress(f"Downloaded {item['name']}")

        # Extraction logic
        qdrant_exe_zip = INSTALL_DIR / 'qdrant-x86_64-pc-windows-msvc.zip'
        qdrant_dist_zip = INSTALL_DIR / 'dist-qdrant.zip'
        data_zip = INSTALL_DIR / 'data.zip'

        if not (qdrant_dir / 'qdrant.exe').exists():
            if not qdrant_exe_zip.exists():
                raise FileNotFoundError("Qdrant zip file missing")
            extract_zip(qdrant_exe_zip, qdrant_dir)

        if not (qdrant_dir / 'static').exists():
            if not qdrant_dist_zip.exists():
                raise FileNotFoundError("Qdrant Web UI zip file missing")
            extract_zip(qdrant_dist_zip, qdrant_dir)
            if (qdrant_dir / 'dist').exists():
                (qdrant_dir / 'dist').rename(qdrant_dir / 'static')

        delete_after_setup = INSTALL_DIR / 'delete_after_setup'
        if not (delete_after_setup / snapshot_name).exists() and not loaded_flag.exists():
            if not data_zip.exists():
                raise FileNotFoundError("Data zip file missing")
            extract_zip(data_zip, delete_after_setup)

        update_progress("Extraction complete")

        if not loaded_flag.exists():
            loaded_flag.touch()
            logger.info("Setup completed successfully")

    except Exception as e:
        logger.critical(f"Setup failed: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    try:
        download_qdrant()
    except KeyboardInterrupt:
        logger.warning("Installation interrupted by user")
        sys.exit(1)