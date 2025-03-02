@echo off
setlocal EnableDelayedExpansion

:: Logging and state file
set "LOG_FILE=%~dp0installation.log"
set "STATE_FILE=%~dp0install_state.txt"
echo [%date% %time%] Starting installation script > "%LOG_FILE%"

:: Progress stages (total weight = 100)
set "STAGES_TOTAL=8" :: Git check, Miniconda download, install, env create, activate, pip install, qdrant install, final run
set "STAGE_WEIGHT=12" :: 100 / 8
set "PROGRESS=0"

:: Function to update progress
:UpdateProgress
set /a "PROGRESS+=!STAGE_WEIGHT!"
echo [%date% %time%] INFO: Progress: !PROGRESS!%% complete >> "%LOG_FILE%"
echo Progress: !PROGRESS!%% complete
exit /b

:: Check state and resume
if exist "%STATE_FILE%" (
    for /f "tokens=*" %%i in (%STATE_FILE%) do set "LAST_STAGE=%%i"
    echo Resuming from stage: !LAST_STAGE! >> "%LOG_FILE%"
) else (
    set "LAST_STAGE=0"
)

:: Stage 1: Check Git
if !LAST_STAGE! LSS 1 (
    call :CheckDependency git "https://git-scm.com/download/win" "Git is required"
    if !ERRORLEVEL! NEQ 0 goto :error_exit
    echo 1 > "%STATE_FILE%"
    call :UpdateProgress
)

cd /D "%~dp0" || goto :error_exit
set "PATH=%PATH%;%SystemRoot%\system32"

:: Stage 2: Download Miniconda
set "INSTALL_DIR=%CD%\installer_files"
if !LAST_STAGE! LSS 2 (
    if not exist "%INSTALL_DIR%" mkdir "%INSTALL_DIR%"
    call :DownloadFile "https://repo.anaconda.com/miniconda/Miniconda3-py310_23.3.1-0-Windows-x86_64.exe" "%INSTALL_DIR%\miniconda_installer.exe" "8a4e8a32e..." :: Add actual hash
    if !ERRORLEVEL! NEQ 0 (
        echo Trying fallback URL...
        call :DownloadFile "https://mirror.example.com/miniconda.exe" "%INSTALL_DIR%\miniconda_installer.exe" "8a4e8a32e..."
        if !ERRORLEVEL! NEQ 0 goto :error_exit
    )
    echo 2 > "%STATE_FILE%"
    call :UpdateProgress
)

:: Stage 3: Install Miniconda
set "CONDA_ROOT_PREFIX=%INSTALL_DIR%\conda"
if !LAST_STAGE! LSS 3 (
    if not exist "%CONDA_ROOT_PREFIX%\_conda.exe" (
        start /wait "" "%INSTALL_DIR%\miniconda_installer.exe" /S /D="%CONDA_ROOT_PREFIX%" 2>>"%LOG_FILE%" || goto :error_exit
    )
    echo 3 > "%STATE_FILE%"
    call :UpdateProgress
)

:: Stage 4: Create Environment
set "INSTALL_ENV_DIR=%INSTALL_DIR%\env"
if !LAST_STAGE! LSS 4 (
    if not exist "%INSTALL_ENV_DIR%" (
        "%CONDA_ROOT_PREFIX%\_conda.exe" create --no-shortcuts -y -k --prefix "%INSTALL_ENV_DIR%" python=3.11 >> "%LOG_FILE%" 2>&1 || goto :error_exit
    )
    echo 4 > "%STATE_FILE%"
    call :UpdateProgress
)

:: Stage 5: Activate Environment
if !LAST_STAGE! LSS 5 (
    call "%CONDA_ROOT_PREFIX%\condabin\conda.bat" activate "%INSTALL_ENV_DIR%" || goto :error_exit
    echo 5 > "%STATE_FILE%"
    call :UpdateProgress
)

:: Stage 6: Install Requests
if !LAST_STAGE! LSS 6 (
    call pip install requests >> "%LOG_FILE%" 2>&1 || goto :error_exit
    echo 6 > "%STATE_FILE%"
    call :UpdateProgress
)

:: Stage 7: Run Qdrant Install
if !LAST_STAGE! LSS 7 (
    call python install_qdrant.py || goto :error_exit
    echo 7 > "%STATE_FILE%"
    call :UpdateProgress
)

:: Stage 8: Final Run
if !LAST_STAGE! LSS 8 (
    call python one_click.py || goto :error_exit
    echo 8 > "%STATE_FILE%"
    call :UpdateProgress
)

del "%STATE_FILE%"
echo [%date% %time%] INFO: Installation completed successfully >> "%LOG_FILE%"
goto :end

:DownloadFile
:: ... (previous download logic with hash check)
exit /b

:CheckDependency
:: ... (previous logic)
exit /b

:error_exit
echo [%date% %time%] ERROR: Failed at stage !LAST_STAGE!. Check %LOG_FILE% >> "%LOG_FILE%"
pause
exit /b 1

:end
pause
exit /b 0