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

# Define the required PyTorch version
TORCH_VERSION = "2.2.1"
TORCHVISION_VERSION = "0.17.1"
TORCHAUDIO_VERSION = "2.2.1"

script_dir = os.getcwd()
conda_env_path = os.path.join(script_dir, "installer_files", "env")

# Command-line flags
cmd_flags_path = os.path.join(script_dir, "CMD_FLAGS.txt")
if os.path.exists(cmd_flags_path):
	with open(cmd_flags_path, 'r') as f:
		CMD_FLAGS = ' '.join(line.strip().rstrip('\\').strip() for line in f if line.strip().rstrip('\\').strip() and not line.strip().startswith('#'))
else:
	CMD_FLAGS = ''

flags = f"{' '.join([flag for flag in sys.argv[1:] if flag != '--update-wizard'])} {CMD_FLAGS}"


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

def install_webui():

	run_cmd(f"mkdir installer_files\\env\\bin", assert_success=True, environment=True)

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


# Install/update the project requirements
# run_cmd("python -m pip install -r requirements.txt --upgrade", assert_success=True, environment=True)
#
#
# print()
# print("There is a good chance that llama-cpp is only working with CPU now.")
# print("We could try to copy a precompiled version into the conda environment")
# print("By feedback it looks like you need to have the Nvidia Cuda Package installed, it has to be 12.2")
# print("If you like to try say Y, we will take a backup so you can rollback if you need later")
# print()
# print()
#
# choice = input("Input> ").upper()
# while choice not in 'YN':
# 	print("Invalid choice. Please try again.")
# 	choice = input("Input> ").upper()
#
# if choice == 'Y':
# 	from distutils.dir_util import copy_tree
# 	import shutil
# 	import os
#
# 	print_big_message("Installing the CUDA runtime libraries.")
# 	run_cmd(f"conda install -y -c \"nvidia/label/cuda-12.2.0\" cuda-runtime", assert_success=True, environment=True)
#
#
#
# 	llama_directory = 'installer_files/env/Lib/site-packages/llama_cpp'
# 	to_directory = 'installer_files/llama_cpp_backup'
# 	compiled_llama_directory = '../llama-cpp_windows/llama_cpp'
#
# 	copy_tree(llama_directory, to_directory)
#
# 	if os.path.exists(llama_directory):
# 		shutil.rmtree(llama_directory)
# 	copy_tree(compiled_llama_directory, llama_directory)
#


install_webui()

launch_webui()