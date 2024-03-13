
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

launch_webui()