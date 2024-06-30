# Copyright 2023 osiworx

# Licensed under the Apache License, Version 2.0 (the "License"); you
# may not use this file except in compliance with the License.  You
# may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied.  See the License for the specific language governing
# permissions and limitations under the License.

# credits go to the oobabooga project where I took most of this files content, thank you very much for your work and inspiration =)


import os
import sys
import platform
import subprocess
import time
import site
import importlib.util
import shutil
import zipfile


# Define the required PyTorch version
TORCH_VERSION = "2.2.1"
TORCHVISION_VERSION = "0.17.1"
TORCHAUDIO_VERSION = "2.2.1"

index_url = os.environ.get('INDEX_URL', "")
python = sys.executable
clip_package = os.environ.get('CLIP_PACKAGE', "git+https://github.com/openai/CLIP.git@a1d071733d7111c9c014f024669f959182114e33")

script_dir = os.getcwd()
conda_env_path = os.path.join(script_dir, "installer_files", "env")

install_dir = os.path.join(script_dir, "installer_files")
cache_dir = os.path.join(script_dir, "installer_cache")


# Command-line flags
cmd_flags_path = os.path.join(script_dir, "CMD_FLAGS.txt")
if os.path.exists(cmd_flags_path):
	with open(cmd_flags_path, 'r') as f:
		CMD_FLAGS = ' '.join(line.strip().rstrip('\\').strip() for line in f if line.strip().rstrip('\\').strip() and not line.strip().startswith('#'))
else:
	CMD_FLAGS = ''

flags = f"{' '.join([flag for flag in sys.argv[1:] if flag != '--update-wizard'])} {CMD_FLAGS}"


# Function to install a package using pip
def install_package(package):
	subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
	import requests
except ImportError:
	print("Requests library not found. Installing...")
	install_package("requests")
	import requests

# Try to import the tqdm module and install it if not present
try:
	from tqdm import tqdm
except ImportError:
	print("tqdm library not found. Installing...")
	install_package("tqdm")
	from tqdm import tqdm

def is_linux():
	return sys.platform.startswith("linux")


def is_windows():
	return sys.platform.startswith("win")


def is_macos():
	return sys.platform.startswith("darwin")


def is_x86_64():
	return platform.machine() == "x86_64"


def cpu_has_avx2():
	try:
		import cpuinfo

		info = cpuinfo.get_cpu_info()
		if 'avx2' in info['flags']:
			return True
		else:
			return False
	except:
		return True


def cpu_has_amx():
	try:
		import cpuinfo

		info = cpuinfo.get_cpu_info()
		if 'amx' in info['flags']:
			return True
		else:
			return False
	except:
		return True


def torch_version():
	site_packages_path = None
	for sitedir in site.getsitepackages():
		if "site-packages" in sitedir and conda_env_path in sitedir:
			site_packages_path = sitedir
			break

	if site_packages_path:
		torch_version_file = open(os.path.join(site_packages_path, 'torch', 'version.py')).read().splitlines()
		torver = [line for line in torch_version_file if line.startswith('__version__')][0].split('__version__ = ')[1].strip("'")
	else:
		from torch import __version__ as torver

	return torver


def update_pytorch():
	print_big_message("Checking for PyTorch updates")

	torver = torch_version()
	is_cuda = '+cu' in torver
	is_cuda118 = '+cu118' in torver  # 2.1.0+cu118
	is_rocm = '+rocm' in torver  # 2.0.1+rocm5.4.2
	is_intel = '+cxx11' in torver  # 2.0.1a0+cxx11.abi
	is_cpu = '+cpu' in torver  # 2.0.1+cpu

	install_pytorch = f"python -m pip install --upgrade torch=={TORCH_VERSION} torchvision=={TORCHVISION_VERSION} torchaudio=={TORCHAUDIO_VERSION} "

	if is_cuda118:
		install_pytorch += "--index-url https://download.pytorch.org/whl/cu118"
	elif is_cuda:
		install_pytorch += "--index-url https://download.pytorch.org/whl/cu121"
	elif is_rocm:
		install_pytorch += "--index-url https://download.pytorch.org/whl/rocm5.6"
	elif is_cpu:
		install_pytorch += "--index-url https://download.pytorch.org/whl/cpu"
	elif is_intel:
		if is_linux():
			install_pytorch = "python -m pip install --upgrade torch==2.1.0a0 torchvision==0.16.0a0 torchaudio==2.1.0a0 intel-extension-for-pytorch==2.1.10+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/"
		else:
			install_pytorch = "python -m pip install --upgrade torch==2.1.0a0 torchvision==0.16.0a0 torchaudio==2.1.0a0 intel-extension-for-pytorch==2.1.10 --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/"

	run_cmd(f"{install_pytorch}", assert_success=True, environment=True)

def clear_cache():
	run_cmd("conda clean -a -y", environment=True)
	run_cmd("python -m pip cache purge", environment=True)


def update_requirements(initial_installation=False):
	# Update PyTorch
	if not initial_installation:
		update_pytorch()

	# Detect the PyTorch version
	torver = torch_version()
	is_cuda = '+cu' in torver
	is_cuda118 = '+cu118' in torver  # 2.1.0+cu118
	is_rocm = '+rocm' in torver  # 2.0.1+rocm5.4.2
	is_intel = '+cxx11' in torver  # 2.0.1a0+cxx11.abi
	is_cpu = '+cpu' in torver  # 2.0.1+cpu

	if is_rocm:
		base_requirements = "requirements_amd" + ("_noavx2" if not cpu_has_avx2() else "") + ".txt"
	elif is_cpu or is_intel:
		base_requirements = "requirements_cpu_only" + ("_noavx2" if not cpu_has_avx2() else "") + ".txt"
	elif is_macos():
		base_requirements = "requirements_apple_" + ("intel" if is_x86_64() else "silicon") + ".txt"
	else:
		base_requirements = "requirements" + ("_noavx2" if not cpu_has_avx2() else "") + ".txt"

	requirements_file = base_requirements

	print_big_message(f"Installing webui requirements from file: {requirements_file}")
	print(f"TORCH: {torver}\n")

	# Prepare the requirements file
	textgen_requirements = open(requirements_file).read().splitlines()
	if is_cuda118:
		textgen_requirements = [req.replace('+cu121', '+cu118').replace('+cu122', '+cu118') for req in textgen_requirements]
	if is_windows() and is_cuda118:  # No flash-attention on Windows for CUDA 11
		textgen_requirements = [req for req in textgen_requirements if 'oobabooga/flash-attention' not in req]

	with open('temp_requirements.txt', 'w') as file:
		file.write('\n'.join(textgen_requirements))

	# Workaround for git+ packages not updating properly.
	git_requirements = [req for req in textgen_requirements if req.startswith("git+")]
	for req in git_requirements:
		url = req.replace("git+", "")
		package_name = url.split("/")[-1].split("@")[0].rstrip(".git")
		run_cmd(f"python -m pip uninstall -y {package_name}", environment=True)
		print(f"Uninstalled {package_name}")

	# Install/update the project requirements
	run_cmd("python -m pip install -r temp_requirements.txt --upgrade", assert_success=True, environment=True)
	os.remove('temp_requirements.txt')

	# Check for '+cu' or '+rocm' in version string to determine if torch uses CUDA or ROCm. Check for pytorch-cuda as well for backwards compatibility
	if not any((is_cuda, is_rocm)) and run_cmd("conda list -f pytorch-cuda | grep pytorch-cuda", environment=True, capture_output=True).returncode == 1:
		clear_cache()
		return

	if not os.path.exists("repositories/"):
		os.mkdir("repositories")

	clear_cache()





def print_big_message(message):
	message = message.strip()
	lines = message.split('\n')
	print("\n\n*******************************************************************")
	for line in lines:
		if line.strip() != '':
			print("*", line)

	print("*******************************************************************\n\n")

def run_cmd(cmd, assert_success=False, environment=False, capture_output=False, env=None):
	# Use the conda environment
	if environment:
		if is_windows():
			conda_bat_path = os.path.join(script_dir, "installer_files", "conda", "condabin", "conda.bat")
			cmd = "\"" + conda_bat_path + "\" activate \"" + conda_env_path + "\" >nul && " + cmd
		else:
			conda_sh_path = os.path.join(script_dir, "installer_files", "conda", "etc", "profile.d", "conda.sh")
			cmd = ". \"" + conda_sh_path + "\" && conda activate \"" + conda_env_path + "\" && " + cmd

	# Run shell commands
	result = subprocess.run(cmd, shell=True, capture_output=capture_output, env=env)

	# Assert the command ran successfully
	if assert_success and result.returncode != 0:
		print("Command '" + cmd + "' failed with exit status code '" + str(result.returncode) )
		print('we will try one more time to see if it will work now')
		result = subprocess.run(cmd, shell=True, capture_output=capture_output, env=env)

		if assert_success and result.returncode != 0:
			print("Command '" + cmd + "' failed with exit status code '" + str(result.returncode) + "'.\n\nExiting now.\nTry running the start/update script again.")
			time.sleep(5)
			sys.exit(1)

	return result


def get_user_choice(question, options_dict):
	print()
	print(question)
	print()

	for key, value in options_dict.items():
		print(f"{key}) {value}")

	print()

	choice = input("Input> ").upper()
	while choice not in options_dict.keys():
		print("Invalid choice. Please try again.")
		choice = input("Input> ").upper()

	return choice

def run(command, desc=None, errdesc=None):
	if desc is not None:
		print(desc)

	result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

	if result.returncode != 0:

		message = f"""{errdesc or 'Error running command'}.
Command: {command}
Error code: {result.returncode}
stdout: {result.stdout.decode(encoding="utf8", errors="ignore") if len(result.stdout)>0 else '<empty>'}
stderr: {result.stderr.decode(encoding="utf8", errors="ignore") if len(result.stderr)>0 else '<empty>'}
"""
		raise RuntimeError(message)

	return result.stdout.decode(encoding="utf8", errors="ignore")
def run_pip(args, desc=None):
	index_url_line = f' --index-url {index_url}' if index_url != '' else ''
	return run(f'"{python}" -m pip {args} --prefer-binary{index_url_line}', desc=f"Installing {desc}", errdesc=f"Couldn't install {desc}")

def is_installed(package):
	try:
		spec = importlib.util.find_spec(package)
	except ModuleNotFoundError:
		return False

	return spec is not None

def install_webui():

	run_cmd(f"mkdir installer_files\\env\\bin", assert_success=True, environment=True)

	if not is_installed("clip"):
		run_pip(f"install {clip_package}", "clip")


	# Ask the user for the GPU vendor
	if "GPU_CHOICE" in os.environ:
		choice = os.environ["GPU_CHOICE"].upper()
		print_big_message(f"Selected GPU choice \"{choice}\" based on the GPU_CHOICE environment variable.")
	else:
		choice = get_user_choice(
			"What is your GPU?",
			{
				'A': 'NVIDIA',
				'B': 'AMD (Linux/MacOS only. Requires ROCm SDK 5.6 on Linux)',
				'C': 'Apple M Series',
				'D': 'Intel Arc (IPEX)',
				'N': 'None (I want to run models in CPU mode)'
			},
		)

	gpu_choice_to_name = {
		"A": "NVIDIA",
		"B": "AMD",
		"C": "APPLE",
		"D": "INTEL",
		"N": "NONE"
	}

	selected_gpu = gpu_choice_to_name[choice]
	use_cuda118 = "N"

	# Write a flag to CMD_FLAGS.txt for CPU mode
	if selected_gpu == "NONE":
		with open(cmd_flags_path, 'r+') as cmd_flags_file:
			if "--cpu" not in cmd_flags_file.read():
				print_big_message("Adding the --cpu flag to CMD_FLAGS.txt.")
				cmd_flags_file.write("\n--cpu\n")

	# Check if the user wants CUDA 11.8
	elif any((is_windows(), is_linux())) and selected_gpu == "NVIDIA":
		if "USE_CUDA118" in os.environ:
			use_cuda118 = "Y" if os.environ.get("USE_CUDA118", "").lower() in ("yes", "y", "true", "1", "t", "on") else "N"
		else:
			print("\nDo you want to use CUDA 11.8 instead of 12.1?\nOnly choose this option if your GPU is very old (Kepler or older).\n\nFor RTX and GTX series GPUs, say \"N\".\nIf unsure, say \"N\".\n")
			use_cuda118 = input("Input (Y/N)> ").upper().strip('"\'').strip()
			while use_cuda118 not in 'YN':
				print("Invalid choice. Please try again.")
				use_cuda118 = input("Input> ").upper().strip('"\'').strip()

		if use_cuda118 == 'Y':
			print("CUDA: 11.8")
		else:
			print("CUDA: 12.1")

	# No PyTorch for AMD on Windows (?)
	elif is_windows() and selected_gpu == "AMD":
		print("PyTorch setup on Windows is not implemented yet. Exiting...")
		sys.exit(1)

	# Find the Pytorch installation command
	install_pytorch = f"python -m pip install torch=={TORCH_VERSION} torchvision=={TORCHVISION_VERSION} torchaudio=={TORCHAUDIO_VERSION} "

	if selected_gpu == "NVIDIA":
		if use_cuda118 == 'Y':
			install_pytorch += "--index-url https://download.pytorch.org/whl/cu118"
		else:
			install_pytorch += "--index-url https://download.pytorch.org/whl/cu121"
	elif selected_gpu == "AMD":
		install_pytorch += "--index-url https://download.pytorch.org/whl/rocm5.6"
	elif selected_gpu in ["APPLE", "NONE"]:
		install_pytorch += "--index-url https://download.pytorch.org/whl/cpu"
	elif selected_gpu == "INTEL":
		if is_linux():
			install_pytorch = "python -m pip install torch==2.1.0a0 torchvision==0.16.0a0 torchaudio==2.1.0a0 intel-extension-for-pytorch==2.1.10+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/"
		else:
			install_pytorch = "python -m pip install torch==2.1.0a0 torchvision==0.16.0a0 torchaudio==2.1.0a0 intel-extension-for-pytorch==2.1.10 --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/"

	# Install Git and then Pytorch
	print_big_message("Installing PyTorch.")
	run_cmd(f"conda install -y -k ninja git && {install_pytorch} && python -m pip install py-cpuinfo==9.0.0", assert_success=True, environment=True)

	if selected_gpu == "INTEL":
		# Install oneAPI dependencies via conda
		print_big_message("Installing Intel oneAPI runtime libraries.")
		run_cmd("conda install -y -c intel dpcpp-cpp-rt=2024.0 mkl-dpcpp=2024.0")
		# Install libuv required by Intel-patched torch
		run_cmd("conda install -y libuv")

	# Install the webui requirements
	update_requirements(initial_installation=True)




def launch_webui():

	run_cmd(f"python pq/prompt_quill_ui_qdrant.py", environment=True)

def extract_zip(zip_path, extract_to):
	with zipfile.ZipFile(zip_path, 'r') as zip_ref:
		zip_ref.extractall(extract_to)

def download_file(url, output_path):
	response = requests.get(url, stream=True)
	total_size = int(response.headers.get('content-length', 0))
	with open(output_path, 'wb') as f, tqdm(
			desc=output_path,
			total=total_size,
			unit='iB',
			unit_scale=True,
			unit_divisor=1024,
	) as bar:
		for data in response.iter_content(chunk_size=1024):
			size = f.write(data)
			bar.update(size)
	return response.status_code

def download_qdrant():

	qdrant_dir = os.path.join(install_dir, 'qdrant')

	if not os.path.exists(qdrant_dir):


		filename = 'qdrant-x86_64-pc-windows-msvc.zip'
		zip_path = os.path.join(cache_dir, filename)
		install_path = os.path.join(install_dir, filename)
		if not os.path.exists(zip_path):
			print("Download Qdrant")
			url = 'https://github.com/qdrant/qdrant/releases/download/v1.9.2/qdrant-x86_64-pc-windows-msvc.zip'
			status_code = download_file(url, install_path)
			if status_code != 200:
				print("\033[101;93m Error: Failed to download qdrant-x86_64-pc-windows-msvc.zip HTTP Status Code: {} \033[0m".format(status_code))
				input("Press Enter to exit...")
				exit(1)
		else:
			shutil.copy(zip_path, install_path)

		filename = 'dist-qdrant.zip'
		zip_path = os.path.join(cache_dir, filename)
		install_path = os.path.join(install_dir, filename)
		if not os.path.exists(zip_path):
			print("Download Qdrant Web UI")
			url = 'https://github.com/qdrant/qdrant-web-ui/releases/download/v0.1.22/dist-qdrant.zip'
			status_code = download_file(url, install_path)
			if status_code != 200:
				print("\033[101;93m Error: Failed to download dist-qdrant.zip HTTP Status Code: {} \033[0m".format(status_code))
				input("Press Enter to exit...")
				exit(1)
		else:
			shutil.copy(zip_path, install_path)

		filename = 'data.zip'
		zip_path = os.path.join(cache_dir, filename)
		install_path = os.path.join(install_dir, filename)
		if not os.path.exists(zip_path):
			print("Download Prompt Quill data")
			url = 'https://civitai.com/api/download/models/567736'
			status_code = download_file(url, install_path)
			if status_code != 200:
				print("\033[101;93m Error: Failed to download prompt quill data HTTP Status Code: {} \033[0m".format(status_code))
				input("Press Enter to exit...")
				exit(1)
		else:
			shutil.copy(zip_path, install_path)


		# Assuming the qdrant executable zip and data zip paths are known and set similarly
		qdrant_exe_zip = os.path.join(install_dir, 'qdrant-x86_64-pc-windows-msvc.zip')
		data_zip = os.path.join(install_dir, 'data.zip')

		print("Extract Qdrant with unzip")
		extract_zip(qdrant_exe_zip, qdrant_dir)

		print("Extract Qdrant web UI with unzip")
		extract_zip(install_path, qdrant_dir)

		print("Rename the dist folder to static")
		os.rename(os.path.join(qdrant_dir, 'dist'), os.path.join(qdrant_dir, 'static'))

		print("Extract data with unzip")
		extract_zip(data_zip, os.path.join(install_dir, 'delete_after_setup'))

		print("Startup Qdrant to upload the data")
		qdrant_executable = os.path.join(qdrant_dir, 'qdrant.exe')
		subprocess.Popen([qdrant_executable, '--disable-telemetry'], cwd=qdrant_dir)

		# Run the Python script and wait for it to complete
		print("Run the check_qdrant_up.py script")
		subprocess.call([sys.executable, 'pq/check_qdrant_up.py'])

		# Upload data using curl
		print("Load data into qdrant, please be patient, this may take a while")
		curl_command = [
			'curl', '-X', 'POST',
			"http://localhost:6333/collections/prompts_large_meta/snapshots/upload?priority=snapshot",
			'-H', "Content-Type:multipart/form-data", '-H', "api-key:",
			'-F', f"snapshot=@{os.path.join(install_dir, 'delete_after_setup', 'prompts_ng_gte-2103298935062809-2024-06-12-06-41-21.snapshot')}"
		]
		subprocess.call(curl_command)

		# Cleanup
		print("Performing cleanup")
		os.remove(os.path.join(install_dir, 'dist-qdrant.zip'))
		os.remove(os.path.join(install_dir, 'qdrant-x86_64-pc-windows-msvc.zip'))
		os.remove(os.path.join(install_dir, 'data.zip'))
		shutil.rmtree(os.path.join(install_dir, 'delete_after_setup'), ignore_errors=True)
		shutil.rmtree(os.path.join(install_dir, 'qdrant', 'snapshots'), ignore_errors=True)

	else:
		print("Startup Qdrant to upload the data")
		qdrant_executable = os.path.join(qdrant_dir, 'qdrant.exe')
		subprocess.Popen([qdrant_executable, '--disable-telemetry'], cwd=qdrant_dir)

		# Run the Python script and wait for it to complete
		print("Run the check_qdrant_up.py script")
		subprocess.call([sys.executable, 'pq/check_qdrant_up.py'])





download_qdrant()

install_webui()

launch_webui()