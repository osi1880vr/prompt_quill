
import os
import sys
import platform
import subprocess



script_dir = os.getcwd()
conda_env_path = os.path.join(script_dir, "installer_files", "env")


def is_linux():
	return sys.platform.startswith("linux")


def is_windows():
	return sys.platform.startswith("win")


def is_macos():
	return sys.platform.startswith("darwin")


def is_x86_64():
	return platform.machine() == "x86_64"

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
		print("Command '" + cmd + "' failed with exit status code '" + str(result.returncode) + "'.\n\nExiting now.\nTry running the start/update script again.")
		sys.exit(1)

	return result


def launch_webui():
	run_cmd(f"python prompt_quill_ui_qdrant.py", environment=True)


# Install/update the project requirements
run_cmd("python -m pip install -r requirements.txt --upgrade", assert_success=True, environment=True)


print()
print("There is a good chance that llama-cpp is only working with CPU now.")
print("We could try to copy a precompiled version into the conda environment")
print("If you like to try say Y, we will take a backup so you can rollback if you need later")
print()
print()

choice = input("Input> ").upper()
while choice not in 'YN':
	print("Invalid choice. Please try again.")
	choice = input("Input> ").upper()

if choice == 'Y':
	from distutils.dir_util import copy_tree
	import shutil
	import os

	print_big_message("Installing the CUDA runtime libraries.")
	run_cmd(f"conda install -y -c \"nvidia/label/cuda-12.2.0\" cuda-runtime", assert_success=True, environment=True)

	run_cmd(f"mkdir installer_files\\env\\bin", assert_success=True, environment=True)

	llama_directory = 'installer_files/env/Lib/site-packages/llama_cpp'
	to_directory = 'installer_files/llama_cpp_backup'
	compiled_llama_directory = '../llama-cpp_windows/llama_cpp'

	copy_tree(llama_directory, to_directory)

	if os.path.exists(llama_directory):
		shutil.rmtree(llama_directory)
	copy_tree(compiled_llama_directory, llama_directory)




launch_webui()