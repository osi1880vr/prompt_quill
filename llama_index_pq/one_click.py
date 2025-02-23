# Copyright 2023 osiworx
# Licensed under the Apache License, Version 2.0
# ... (keeping the original license header)

import os
import sys
import platform
import subprocess
import time
import site
import importlib.util
import shutil
import logging
from typing import Optional, Dict
from pathlib import Path

# Configure logging
logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s - %(levelname)s - %(message)s',
	handlers=[
		logging.FileHandler('installation.log'),
		logging.StreamHandler()
	]
)
logger = logging.getLogger(__name__)

# Constants
TORCH_VERSION = "2.4.1"
TORCHVISION_VERSION = "0.19.1"
TORCHAUDIO_VERSION = "2.4.1"

INDEX_URL = os.environ.get('INDEX_URL', "")
CLIP_PACKAGE = os.environ.get('CLIP_PACKAGE', "git+https://github.com/openai/CLIP.git@a1d071733d7111c9c014f024669f959182114e33")
PYTHON = sys.executable

# Paths
BASE_DIR = Path.cwd()
INSTALL_DIR = BASE_DIR / "installer_files"
CACHE_DIR = BASE_DIR / "installer_cache"
CONDA_ENV_PATH = INSTALL_DIR / "env"
CMD_FLAGS_PATH = BASE_DIR / "CMD_FLAGS.txt"

# Load command-line flags
CMD_FLAGS = ""
if CMD_FLAGS_PATH.exists():
	with open(CMD_FLAGS_PATH, 'r') as f:
		CMD_FLAGS = ' '.join(line.strip().rstrip('\\').strip()
							 for line in f
							 if line.strip().rstrip('\\').strip() and not line.strip().startswith('#'))

FLAGS = f"{' '.join([flag for flag in sys.argv[1:] if flag != '--update-wizard'])} {CMD_FLAGS}"

def install_package(package: str) -> bool:
	"""Install a package using pip."""
	try:
		subprocess.check_call([PYTHON, "-m", "pip", "install", package])
		return True
	except subprocess.CalledProcessError as e:
		logger.error(f"Failed to install {package}: {str(e)}")
		return False

# Lazy imports
for module, package in [("requests", "requests"), ("tqdm", "tqdm")]:
	if not importlib.util.find_spec(module):
		logger.warning(f"{module} not found. Installing...")
		if not install_package(package):
			logger.critical(f"Failed to install {package}. Aborting.")
			sys.exit(1)
import requests
from tqdm import tqdm

# Platform checks
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

def get_torch_version() -> str:
	"""Get installed PyTorch version."""
	try:
		site_packages = next((p for p in site.getsitepackages() if "site-packages" in p and str(CONDA_ENV_PATH) in p), None)
		if site_packages:
			with open(Path(site_packages) / 'torch' / 'version.py') as f:
				return [line.split('__version__ = ')[1].strip("'") for line in f if line.startswith('__version__')][0]
		import torch
		return torch.__version__
	except Exception as e:
		logger.error(f"Failed to get torch version: {str(e)}")
		raise

def run_cmd(cmd: str, assert_success: bool = False, environment: bool = False, capture_output: bool = False) -> subprocess.CompletedProcess:
	"""Run a shell command with optional environment activation."""
	if environment:
		if is_platform("win"):
			cmd = f'"{INSTALL_DIR / "conda" / "condabin" / "conda.bat"}" activate "{CONDA_ENV_PATH}" >nul && {cmd}'
		else:
			cmd = f'. "{INSTALL_DIR / "conda" / "etc" / "profile.d" / "conda.sh"}" && conda activate "{CONDA_ENV_PATH}" && {cmd}'

	try:
		result = subprocess.run(cmd, shell=True, capture_output=capture_output, text=True)
		if assert_success and result.returncode != 0:
			logger.error(f"Command failed: {cmd}\nExit code: {result.returncode}\nOutput: {result.stdout}\nError: {result.stderr}")
			time.sleep(2)  # Give chance to retry
			result = subprocess.run(cmd, shell=True, capture_output=capture_output, text=True)
			if result.returncode != 0:
				logger.critical("Command failed after retry. Exiting.")
				sys.exit(1)
		return result
	except Exception as e:
		logger.error(f"Command execution failed: {str(e)}", exc_info=True)
		raise

def update_pytorch() -> None:
	"""Update PyTorch to the specified version."""
	logger.info("Checking PyTorch updates")
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
		cmd = "python -m pip install --upgrade torch==2.1.0a0 torchvision==0.16.0a0 torchaudio==2.1.0a0 intel-extension-for-pytorch==2.1.10" + \
			  ("+xpu" if is_platform("linux") else "") + " --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/"

	run_cmd(cmd, assert_success=True, environment=True)

def update_requirements(initial_install: bool = False) -> None:
	"""Update project requirements based on platform and hardware."""
	if not initial_install:
		update_pytorch()

	torver = get_torch_version()
	variants = {'cuda': '+cu' in torver, 'cuda118': '+cu118' in torver, 'rocm': '+rocm' in torver,
				'intel': '+cxx11' in torver, 'cpu': '+cpu' in torver}

	req_file = ("requirements_amd" if variants['rocm'] else
				"requirements_cpu_only" if variants['cpu'] or variants['intel'] else
				f"requirements_apple_{'intel' if is_x86_64() else 'silicon'}" if is_platform("darwin") else
				"requirements") + ("_noavx2" if not cpu_has_feature('avx2') else "") + ".txt"

	logger.info(f"Installing requirements from {req_file}")
	reqs = open(req_file).read().splitlines()
	if variants['cuda118']:
		reqs = [r.replace('+cu121', '+cu118').replace('+cu122', '+cu118') for r in reqs]
	if is_platform("win") and variants['cuda118']:
		reqs = [r for r in reqs if 'oobabooga/flash-attention' not in r]

	temp_reqs = BASE_DIR / 'temp_requirements.txt'
	with open(temp_reqs, 'w') as f:
		f.write('\n'.join(reqs))

	for req in [r for r in reqs if r.startswith("git+")]:
		pkg = req.split("/")[-1].split("@")[0].rstrip(".git")
		run_cmd(f"python -m pip uninstall -y {pkg}", environment=True)
		logger.info(f"Uninstalled {pkg} for fresh install")

	run_cmd(f"python -m pip install -r {temp_reqs} --upgrade", assert_success=True, environment=True)
	temp_reqs.unlink()

def install_webui() -> None:
	"""Install web UI dependencies."""
	stages = 3  # PyTorch, Requirements, Cleanup
	stage_weight = 33.33
	progress = 0

	def update_progress(stage_name: str):
		nonlocal progress
		progress += stage_weight
		logger.info(f"Web UI Install Progress: {progress:.1f}% - {stage_name}")
		print(f"Web UI Install Progress: {progress:.1f}%")

	"""Install web UI dependencies."""
	if not (CONDA_ENV_PATH / 'bin').exists():
		(CONDA_ENV_PATH / 'bin').mkdir(parents=True)

	try:
		import torch
	except ImportError:
		gpu_choices = {'A': 'NVIDIA', 'B': 'AMD', 'C': 'APPLE', 'D': 'INTEL', 'N': 'NONE'}
		choice = os.environ.get("GPU_CHOICE", "").upper() or get_user_choice(
			"What is your GPU?", {k: f"{v} ({'ROCm required' if k == 'B' else 'CPU mode' if k == 'N' else 'IPEX' if k == 'D' else ''})"
								  for k, v in gpu_choices.items()}
		)
		selected_gpu = gpu_choices[choice]

		if selected_gpu == "NONE" and CMD_FLAGS_PATH.exists():
			with open(CMD_FLAGS_PATH, 'a') as f:
				if "--cpu" not in f.read():
					f.write("\n--cpu\n")
					logger.info("Added --cpu flag to CMD_FLAGS.txt")

		use_cuda118 = "N"
		if selected_gpu == "NVIDIA" and any(is_platform(p) for p in ["win", "linux"]):
			use_cuda118 = os.environ.get("USE_CUDA118", "N").upper()
			if use_cuda118 not in ['Y', 'N']:
				use_cuda118 = input("Use CUDA 11.8 instead of 12.1? (Y/N, recommended N for RTX/GTX): ").upper().strip() or "N"

		cmd = f"python -m pip install torch=={TORCH_VERSION} torchvision=={TORCHVISION_VERSION} torchaudio=={TORCHAUDIO_VERSION}"
		if selected_gpu == "NVIDIA":
			cmd += " --index-url https://download.pytorch.org/whl/cu118" if use_cuda118 == 'Y' else " --index-url https://download.pytorch.org/whl/cu121"
		elif selected_gpu == "AMD":
			cmd += " --index-url https://download.pytorch.org/whl/rocm5.6"
		elif selected_gpu in ["APPLE", "NONE"]:
			cmd += " --index-url https://download.pytorch.org/whl/cpu"
		elif selected_gpu == "INTEL":
			cmd = f"python -m pip install torch==2.1.0a0 torchvision==0.16.0a0 torchaudio==2.1.0a0 intel-extension-for-pytorch==2.1.10{' +xpu' if is_platform('linux') else ''} --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/"

		logger.info("Installing PyTorch and dependencies")
		run_cmd(f"conda install -y -k ninja git && {cmd} && python -m pip install py-cpuinfo==9.0.0", assert_success=True, environment=True)

		if selected_gpu == "INTEL":
			run_cmd("conda install -y -c intel dpcpp-cpp-rt=2024.0 mkl-dpcpp=2024.0", assert_success=True, environment=True)
			run_cmd("conda install -y libuv", assert_success=True, environment=True)

	update_requirements(initial_install=True)
	if not importlib.util.find_spec("clip"):
		run_cmd(f"python -m pip install {CLIP_PACKAGE}", assert_success=True, environment=True)

	update_progress("PyTorch installed")

	update_requirements(initial_install=True)
	update_progress("Requirements updated")

	if not importlib.util.find_spec("clip"):
		run_cmd(f"python -m pip install {CLIP_PACKAGE}", assert_success=True, environment=True)
	update_progress("Web UI setup complete")


def get_available_space(path: Path) -> int:
	"""Get available disk space in bytes."""
	try:
		stat = os.statvfs(str(path)) if os.name == 'posix' else shutil.disk_usage(str(path))
		return stat.f_bavail * stat.f_frsize if os.name == 'posix' else stat.free
	except Exception as e:
		logger.error(f"Failed to check disk space: {str(e)}")
		return float('inf')  # Assume infinite space if check fails

def launch_webui() -> None:
	"""Launch the web UI."""
	min_space_gb = 1  # Adjust based on web UI needs
	if get_available_space(BASE_DIR) < min_space_gb * (1024**3):
		logger.error(f"Insufficient disk space for web UI: {min_space_gb}GB required")
		sys.exit(1)
	logger.info("Launching Prompt Quill UI")
	logger.info("Launching Prompt Quill UI (100% complete)")
	print("Process Complete: 100% - Launching UI")
	run_cmd("python pq/prompt_quill_ui_qdrant.py", environment=True)

def cleanup_qdrant_data() -> None:
	"""Clean up Qdrant installation files."""
	logger.info("Performing cleanup")
	for file in ['dist-qdrant.zip', 'qdrant-x86_64-pc-windows-msvc.zip', 'data.zip']:
		path = INSTALL_DIR / file
		if path.exists():
			path.unlink()
	for dir in [INSTALL_DIR / 'delete_after_setup', INSTALL_DIR / 'qdrant' / 'snapshots']:
		shutil.rmtree(dir, ignore_errors=True)

def get_user_choice(question: str, options: Dict[str, str]) -> str:
	"""Get user input with validation."""
	print(f"\n{question}\n")
	for k, v in options.items():
		print(f"{k}) {v}")
	choice = input("\nInput> ").upper()
	while choice not in options:
		print("Invalid choice. Try again.")
		choice = input("Input> ").upper()
	return choice

if __name__ == "__main__":
	try:
		cleanup_qdrant_data()
		install_webui()
		launch_webui()
	except KeyboardInterrupt:
		logger.warning("Interrupted by user")
		sys.exit(1)
	except Exception as e:
		logger.critical("Execution failed", exc_info=True)
		sys.exit(1)