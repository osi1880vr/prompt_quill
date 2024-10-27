@echo off
REM Change to the directory where this script is located
cd /D "%~dp0"

REM Define relative paths based on the current directory
set INSTALL_DIR=%cd%\installer_files
set CONDA_ROOT_PREFIX=%INSTALL_DIR%\conda
set INSTALL_ENV_DIR=%INSTALL_DIR%\env
set BASE_DIR=%cd%

REM Step 1: Pull the latest code from Git
echo Pulling latest changes from the git repository...
git pull
if errorlevel 1 (
    echo Git pull failed. Exiting.
    exit /b 1
)

REM Step 2: Activate the Conda environment
echo Activating Conda environment in "%INSTALL_ENV_DIR%"...
call "%CONDA_ROOT_PREFIX%\condabin\conda.bat" activate "%INSTALL_ENV_DIR%" || (
    echo. && echo Miniconda hook not found. Exiting. && exit /b 1
)

REM Step 3: Install or upgrade dependencies from requirements.txt
echo Installing/upgrading dependencies from requirements.txt...
pip install --upgrade -r "updates.txt"
if errorlevel 1 (
    echo Failed to install requirements. Exiting.
    exit /b 1
)

echo Update complete.
pause
exit