# Copyright 2023 osiworx
# Licensed under the Apache License, Version 2.0
# ... (keeping your original license header)

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import site
import subprocess
import sys
import time
import importlib.util
import logging
from pathlib import Path
from typing import Dict, Optional

# Configure logging - console only
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Constants
TORCH_VERSION = "2.4.1"
TORCHVISION_VERSION = "0.19.1"
TORCHAUDIO_VERSION = "2.4.1"
CLIP_PACKAGE = os.environ.get('CLIP_PACKAGE', "git+https://github.com/openai/CLIP.git@a1d071733d7111c9c014f024669f959182114e33")
PYTHON = sys.executable

# Paths
BASE_DIR = Path.cwd()
INSTALL_DIR = BASE_DIR / "installer_files"
CACHE_DIR = BASE_DIR / "installer_cache"
CONDA_ENV_PATH = INSTALL_DIR / "env"
CMD_FLAGS_PATH = BASE_DIR / "CMD_FLAGS.txt"
STATE_FILE = BASE_DIR / ".installer_state.json"

# Load command-line flags
CMD_FLAGS = ""
if CMD_FLAGS_PATH.exists():
    with open(CMD_FLAGS_PATH, 'r') as f:
        CMD_FLAGS = ' '.join(line.strip().rstrip('\\').strip()
                             for line in f
                             if line.strip().rstrip('\\').strip() and not line.strip().startswith('#'))
FLAGS = f"{' '.join([flag for flag in sys.argv[1:] if flag != '--update-wizard'])} {CMD_FLAGS}"


# Near the top, after imports and constants
def get_extensions_names() -> list:
    """Return list of extension names with a requirements.txt, excluding examples."""
    extensions_dir = BASE_DIR / "extensions"
    if not extensions_dir.exists():
        return []
    return [folder.name for folder in extensions_dir.iterdir()
            if folder.is_dir() and not folder.name.startswith('__') and folder.name != 'examples'
            and (folder / "requirements.txt").exists()]

def install_extensions_requirements() -> None:
    """Install requirements for all extensions."""
    extensions = get_extensions_names()
    if not extensions:
        logger.info("No extensions found to install.")
        return
    print_big_message("Installing extensions requirements.\nSome may fail on Windows—don’t worry, they won’t affect the main app.")
    for i, extension in enumerate(extensions, 1):
        print(f"\n\n--- [{i}/{len(extensions)}]: {extension}\n\n")
        extension_req_path = BASE_DIR / "extensions" / extension / "requirements.txt"
        run_cmd(f"python -m pip install -r {extension_req_path} --upgrade", assert_success=False, environment=True)



# Signal handler for clean exit
def signal_handler(sig, frame):
    logger.info("Received interrupt signal. Exiting cleanly.")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Platform and hardware checks
def is_platform(check: str) -> bool:
    return sys.platform.startswith(check)

def is_x86_64() -> bool:
    return platform.machine() == "x86_64"

def cpu_has_feature(feature: str) -> bool:
    try:
        import cpuinfo
        return feature in cpuinfo.get_cpu_info().get('flags', [])
    except Exception:
        logger.warning(f"Could not check CPU feature {feature}. Assuming supported.")
        return True

# Lazy imports
for module, package in [("requests", "requests"), ("tqdm", "tqdm"), ("cpuinfo", "py-cpuinfo==9.0.0")]:
    if not importlib.util.find_spec(module):
        logger.warning(f"{module} not found. Installing...")
        subprocess.check_call([PYTHON, "-m", "pip", "install", package])

import requests
from tqdm import tqdm

# Utility functions
def print_big_message(message: str):
    lines = message.strip().split('\n')
    print("\n\n*******************************************************************")
    for line in lines:
        print("*", line)
    print("*******************************************************************\n\n")

def calculate_file_hash(file_path: Path) -> str:
    if file_path.is_file():
        with open(file_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    return ''

def get_current_commit() -> str:
    try:
        return subprocess.check_output("git rev-parse HEAD", shell=True, text=True).strip()
    except subprocess.CalledProcessError:
        return ""

def run_cmd(cmd: str, assert_success: bool = False, environment: bool = False, capture_output: bool = False) -> subprocess.CompletedProcess:
    if environment:
        if is_platform("win"):
            cmd = f'"{INSTALL_DIR / "conda" / "condabin" / "conda.bat"}" activate "{CONDA_ENV_PATH}" >nul && {cmd}'
        else:
            cmd = f'. "{INSTALL_DIR / "conda" / "etc" / "profile.d" / "conda.sh"}" && conda activate "{CONDA_ENV_PATH}" && {cmd}'

    executable = None if is_platform("win") else "bash"
    try:
        result = subprocess.run(cmd, shell=True, capture_output=capture_output, text=True, executable=executable)
        if assert_success and result.returncode != 0:
            logger.error(f"Command failed: {cmd}\nExit code: {result.returncode}\nOutput: {result.stdout}\nError: {result.stderr}")
            time.sleep(2)  # Retry once
            result = subprocess.run(cmd, shell=True, capture_output=capture_output, text=True, executable=executable)
            if result.returncode != 0:
                logger.critical("Command failed after retry. Exiting.")
                sys.exit(1)
        return result
    except Exception as e:
        logger.error(f"Command execution failed: {str(e)}", exc_info=True)
        raise

def get_torch_version() -> str:
    try:
        site_packages = next((p for p in site.getsitepackages() if "site-packages" in p and str(CONDA_ENV_PATH) in p), None)
        if site_packages:
            with open(Path(site_packages) / 'torch' / 'version.py') as f:
                return [line.split('__version__ = ')[1].strip("'") for line in f if line.startswith('__version__')][0]
        import torch
        return torch.__version__
    except Exception:
        return "unknown"

def load_state() -> Dict:
    if STATE_FILE.exists():
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_state(state: Dict):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f)

# Core installation functions
def update_pytorch() -> None:
    print_big_message("Checking for PyTorch updates.")
    torver = get_torch_version()
    variants = {'cuda': '+cu' in torver, 'cuda118': '+cu118' in torver, 'rocm': '+rocm' in torver,
                'intel': '+cxx11' in torver, 'cpu': '+cpu' in torver}

    cmd = f"python -m pip install --upgrade torch=={TORCH_VERSION} torchvision=={TORCHVISION_VERSION} torchaudio=={TORCHAUDIO_VERSION}"
    if variants['cuda118']:
        cmd += " --index-url https://download.pytorch.org/whl/cu118"
    elif variants['cuda']:
        cmd += " --index-url https://download.pytorch.org/whl/cu121"
    elif variants['rocm']:
        cmd += " --index-url https://download.pytorch.org/whl/rocm5.6"
    elif variants['cpu']:
        cmd += " --index-url https://download.pytorch.org/whl/cpu"
    elif variants['intel']:
        cmd += " --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/" + \
               (" +xpu" if is_platform("linux") else "")

    run_cmd(cmd, assert_success=True, environment=True)

def get_requirements_file() -> str:
    torver = get_torch_version()
    variants = {'cuda': '+cu' in torver, 'cuda118': '+cu118' in torver, 'rocm': '+rocm' in torver,
                'intel': '+cxx11' in torver, 'cpu': '+cpu' in torver}
    return ("requirements_amd" if variants['rocm'] else
            "requirements_cpu_only" if variants['cpu'] or variants['intel'] else
            f"requirements_apple_{'intel' if is_x86_64() else 'silicon'}" if is_platform("darwin") else
            "requirements") + ("_noavx2" if not cpu_has_feature('avx2') else "") + ".txt"


def install_webui() -> None:
    stages = 4  # PyTorch, Requirements, CLIP, Extensions+Cleanup
    stage_weight = 25.0
    progress = 0

    def update_progress(stage_name: str):
        nonlocal progress
        progress += stage_weight
        logger.info(f"Web UI Install Progress: {progress:.1f}% - {stage_name}")
        print(f"Web UI Install Progress: {progress:.1f}%")

    if not (CONDA_ENV_PATH / 'bin').exists():
        (CONDA_ENV_PATH / 'bin').mkdir(parents=True)

    try:
        import torch
    except ImportError:
        gpu_choices = {
            'A': 'NVIDIA - CUDA 12.1 (recommended)',
            'B': 'NVIDIA - CUDA 11.8 (legacy GPUs)',
            'C': 'AMD - ROCm 5.6',
            'D': 'Apple M Series',
            'E': 'Intel Arc (beta)',
            'N': 'CPU mode'
        }
        choice = os.environ.get("GPU_CHOICE", "").upper() or get_user_choice("What is your GPU?", gpu_choices)
        selected_gpu = {'A': 'NVIDIA', 'B': 'NVIDIA', 'C': 'AMD', 'D': 'APPLE', 'E': 'INTEL', 'N': 'NONE'}[choice]
        use_cuda118 = (choice == 'B')

        with open(BASE_DIR / "gpu_choice.txt", 'w') as f:
            f.write(choice)

        if selected_gpu == "NONE" and CMD_FLAGS_PATH.exists():
            with open(CMD_FLAGS_PATH, 'a') as f:
                if "--cpu" not in f.read():
                    f.write("\n--cpu\n")
                    logger.info("Added --cpu flag to CMD_FLAGS.txt")

        cmd = f"python -m pip install torch=={TORCH_VERSION} torchvision=={TORCHVISION_VERSION} torchaudio=={TORCHAUDIO_VERSION}"
        if selected_gpu == "NVIDIA":
            cmd += " --index-url https://download.pytorch.org/whl/cu118" if use_cuda118 else " --index-url https://download.pytorch.org/whl/cu121"
            print_big_message("Ensure NVIDIA drivers, CuDNN, and CUDA are installed.")
        elif selected_gpu == "AMD":
            cmd += " --index-url https://download.pytorch.org/whl/rocm5.6"
        elif selected_gpu in ["APPLE", "NONE"]:
            cmd += " --index-url https://download.pytorch.org/whl/cpu"
        elif selected_gpu == "INTEL":
            cmd = f"python -m pip install torch==2.1.0a0 torchvision==0.16.0a0 torchaudio==2.1.0a0 intel-extension-for-pytorch==2.1.10{' +xpu' if is_platform('linux') else ''} --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/"

        print_big_message("Installing PyTorch and dependencies")
        run_cmd(f"conda install -y -k ninja git && {cmd}", assert_success=True, environment=True)

        if selected_gpu == "INTEL":
            run_cmd("conda install -y -c intel dpcpp-cpp-rt=2024.0 mkl-dpcpp=2024.0", assert_success=True, environment=True)
            run_cmd("conda install -y libuv", assert_success=True, environment=True)

    update_progress("PyTorch installed")
    update_requirements(initial_install=True, cuda_choice=choice)
    update_progress("Requirements updated")

    if not importlib.util.find_spec("clip"):
        run_cmd(f"python -m pip install {CLIP_PACKAGE}", assert_success=True, environment=True)
    update_progress("CLIP installed")

    if os.environ.get("INSTALL_EXTENSIONS", "").lower() in ("yes", "y", "true", "1", "t", "on"):
        install_extensions_requirements()
    cleanup_qdrant_data()
    update_progress("Setup complete")


def update_requirements(initial_install: bool = False, pull: bool = True, cuda_choice: str = 'A') -> None:
    state = load_state()
    current_commit = get_current_commit()
    wheels_changed = state.get('wheels_changed', False) or state.get('last_installed_commit') != current_commit

    if pull and current_commit:
        req_file = get_requirements_file()
        before_whl = [line for line in open(req_file).read().splitlines() if '.whl' in line] if Path(req_file).exists() else []
        print_big_message("Updating repository with 'git pull'")

        files_to_check = ['pq/prompt_quill_ui_qdrant.py']
        before_hashes = {f: calculate_file_hash(BASE_DIR / f) for f in files_to_check}

        run_cmd("git pull --autostash", assert_success=True, environment=True)

        after_hashes = {f: calculate_file_hash(BASE_DIR / f) for f in files_to_check}
        after_whl = [line for line in open(req_file).read().splitlines() if '.whl' in line] if Path(req_file).exists() else []
        wheels_changed = wheels_changed or (before_whl != after_whl)

        for f in files_to_check:
            if before_hashes[f] != after_hashes[f]:
                print_big_message(f"File '{f}' updated. Please rerun the script.")
                save_state({'last_installed_commit': current_commit, 'wheels_changed': wheels_changed})
                sys.exit(1)

    save_state({'last_installed_commit': current_commit, 'wheels_changed': False})

    if not initial_install:
        update_pytorch()

    req_file = get_requirements_file()
    print_big_message(f"Installing requirements from {req_file}")
    reqs = open(req_file).read().splitlines()

    cuda_version = "12.1" if cuda_choice == 'A' else "11.8" if cuda_choice == 'B' else None
    if cuda_version:
        reqs = [r.replace('CUDA_VERSION', cuda_version) for r in reqs]
    else:
        reqs = [r for r in reqs if 'cudatoolkit' not in r]

    torver = get_torch_version()
    if '+cu118' in torver:
        reqs = [r.replace('+cu121', '+cu118').replace('+cu122', '+cu118') for r in reqs]
    if is_platform("win") and '+cu118' in torver:
        reqs = [r for r in reqs if 'flash-attention' not in r.lower()]

    if not initial_install and not wheels_changed:
        reqs = [r for r in reqs if '.whl' not in r]

    temp_reqs = BASE_DIR / 'temp_requirements.txt'
    with open(temp_reqs, 'w') as f:
        f.write('\n'.join(reqs))

    for req in [r for r in reqs if r.startswith("git+")]:
        pkg = req.split("/")[-1].split("@")[0].rstrip(".git")
        run_cmd(f"python -m pip uninstall -y {pkg}", environment=True)
        logger.info(f"Uninstalled {pkg} for fresh install")

    conda_reqs = [r for r in reqs if 'cudatoolkit' in r]
    if conda_reqs:
        run_cmd(f"conda install -y {' '.join(conda_reqs)}", assert_success=True, environment=True)

    run_cmd(f"python -m pip install -r {temp_reqs} --upgrade", assert_success=True, environment=True)
    temp_reqs.unlink()

    if os.environ.get("INSTALL_EXTENSIONS", "").lower() in ("yes", "y", "true", "1", "t", "on"):
        install_extensions_requirements()


def get_available_space(path: Path) -> int:
    try:
        stat = os.statvfs(str(path)) if os.name == 'posix' else shutil.disk_usage(str(path))
        return stat.f_bavail * stat.f_frsize if os.name == 'posix' else stat.free
    except Exception:
        return float('inf')


def launch_webui() -> None:
    min_space_gb = 1
    if get_available_space(BASE_DIR) < min_space_gb * (1024**3):
        logger.error(f"Insufficient disk space: {min_space_gb}GB required")
        sys.exit(1)
    print_big_message("Launching Prompt Quill UI")
    run_cmd(f"python pq/prompt_quill_ui_qdrant.py {FLAGS}", environment=True)


def cleanup_qdrant_data() -> None:
    logger.info("Performing cleanup")
    for file in ['dist-qdrant.zip', 'qdrant-x86_64-pc-windows-msvc.zip', 'data.zip']:
        (INSTALL_DIR / file).unlink(missing_ok=True)
    for dir in [INSTALL_DIR / 'delete_after_setup', INSTALL_DIR / 'qdrant' / 'snapshots']:
        shutil.rmtree(dir, ignore_errors=True)


def get_user_choice(question: str, options: Dict[str, str]) -> str:
    print_big_message(question)
    for k, v in options.items():
        print(f"{k}) {v}")
    choice = input("\nInput> ").upper()
    while choice not in options:
        print("Invalid choice. Try again.")
        choice = input("Input> ").upper()
    return choice


def update_wizard():
    while True:
        choice = get_user_choice(
            "What would you like to do?",
            {'A': 'Update Prompt Quill UI', 'B': 'Update PyTorch', 'C': 'Reset repository (if Git)', 'N': 'Exit'}
        )
        if choice == 'A':
            update_requirements()
        elif choice == 'B':
            update_pytorch()
        elif choice == 'C' and get_current_commit():
            run_cmd("git reset --hard", assert_success=True, environment=True)
        elif choice == 'N':
            sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--update-wizard', action='store_true', help='Launch update wizard.')
    args, _ = parser.parse_known_args()

    try:
        if args.update_wizard:
            update_wizard()
        else:
            install_webui()
            if os.environ.get("LAUNCH_AFTER_INSTALL", "yes").lower() not in ("no", "n", "false", "0", "f", "off"):
                launch_webui()
            else:
                print_big_message("Exiting due to LAUNCH_AFTER_INSTALL setting.")
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.critical("Execution failed", exc_info=True)
        sys.exit(1)