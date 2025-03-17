@echo off
setlocal EnableDelayedExpansion

set "LOG_FILE=%~dp0installation.log"
set "STATE_FILE=%~dp0install_state.txt"
echo [%date% %time%] Starting installation script > "%LOG_FILE%"

goto :Main

:UpdateProgress
echo [%date% %time%] INFO: Entering UpdateProgress >> "%LOG_FILE%"
set /a "PROGRESS+=!STAGE_WEIGHT!"
echo [%date% %time%] INFO: Progress: !PROGRESS!%% complete >> "%LOG_FILE%"
echo Progress: !PROGRESS!%% complete
echo [%date% %time%] INFO: Exiting UpdateProgress >> "%LOG_FILE%"
exit /b 0

:Main
set "STAGE_WEIGHT=12"
set "PROGRESS=0"

if exist "%STATE_FILE%" (
    for /f "tokens=*" %%i in ('type "%STATE_FILE%"') do set "LAST_STAGE=%%i"
    echo [%date% %time%] INFO: Resuming from stage: !LAST_STAGE! >> "%LOG_FILE%"
    set /a "PROGRESS=(!LAST_STAGE! - 1) * !STAGE_WEIGHT!"
    echo [%date% %time%] INFO: Initial progress set to !PROGRESS!%% >> "%LOG_FILE%"
    :: Jump to the correct stage
    if !LAST_STAGE! EQU 1 goto :Stage2
    if !LAST_STAGE! EQU 2 goto :Stage3
    if !LAST_STAGE! EQU 3 goto :Stage4
    if !LAST_STAGE! EQU 4 goto :Stage5
    if !LAST_STAGE! EQU 5 goto :Stage6
    if !LAST_STAGE! EQU 6 goto :Stage7
    if !LAST_STAGE! EQU 7 goto :Stage8
    if !LAST_STAGE! GEQ 8 goto :end
) else (
    set "LAST_STAGE=0"
    echo [%date% %time%] INFO: Fresh start >> "%LOG_FILE%"
)

:: Stage 1: Check Git
if !LAST_STAGE! LEQ 1 (
    echo [%date% %time%] INFO: Stage 1 - Checking Git >> "%LOG_FILE%"
    where git >nul 2>&1 || (
        echo [%date% %time%] ERROR: Git not found >> "%LOG_FILE%"
        goto :error_exit
    )
    echo 1 > "%STATE_FILE%"
    call :UpdateProgress
    goto :Stage2
)

:Stage2
if !LAST_STAGE! LEQ 2 (
    echo [%date% %time%] INFO: Stage 2 - Downloading Miniconda >> "%LOG_FILE%"
    set "INSTALL_DIR=%CD%\installer_files"
    if not exist "!INSTALL_DIR!" mkdir "!INSTALL_DIR!" || (
        echo [%date% %time%] ERROR: Failed to create !INSTALL_DIR! >> "%LOG_FILE%"
        goto :error_exit
    )
    set "MINICONDA_URL=https://repo.anaconda.com/miniconda/Miniconda3-py310_23.3.1-0-Windows-x86_64.exe"
    set "MINICONDA_EXE=!INSTALL_DIR!\miniconda_installer.exe"
    if exist "!MINICONDA_EXE!" (
        echo [%date% %time%] INFO: Miniconda installer already exists at !MINICONDA_EXE!, skipping download >> "%LOG_FILE%"
    ) else (
        echo [%date% %time%] INFO: Downloading Miniconda from !MINICONDA_URL! >> "%LOG_FILE%"
        powershell -Command "Invoke-WebRequest -Uri '!MINICONDA_URL!' -OutFile '!MINICONDA_EXE!'" || (
            echo [%date% %time%] ERROR: Failed to download Miniconda >> "%LOG_FILE%"
            goto :error_exit
        )
    )
    echo 2 > "%STATE_FILE%"
    call :UpdateProgress
    goto :Stage3
)

:Stage3
if !LAST_STAGE! LEQ 3 (
    echo [%date% %time%] INFO: Stage 3 - Installing Miniconda >> "%LOG_FILE%"
    set "CONDA_ROOT_PREFIX=%INSTALL_DIR%\conda"
    start /wait "" "!MINICONDA_EXE!" /InstallationType=JustMe /NoShortcuts=1 /AddToPath=0 /RegisterPython=0 /NoRegistry=1 /S /D=!CONDA_ROOT_PREFIX!
    set "CONDA_CMD=!CONDA_ROOT_PREFIX!\_conda.exe"
    echo 3 > "%STATE_FILE%"
    call :UpdateProgress
    goto :Stage4
)

:Stage4
if !LAST_STAGE! LEQ 4 (
    echo [%date% %time%] INFO: Stage 4 - Creating environment >> "%LOG_FILE%"
    set "INSTALL_ENV_DIR=%INSTALL_DIR%\env"
    "!CONDA_CMD!" create --no-shortcuts -y -k --prefix "!INSTALL_ENV_DIR!" python=3.11
    echo 4 > "%STATE_FILE%"
    call :UpdateProgress
    goto :Stage5
)

:Stage5
if !LAST_STAGE! LEQ 5 (
    echo [%date% %time%] INFO: Stage 5 - Activating environment >> "%LOG_FILE%"
    set "INSTALL_ENV_DIR=%INSTALL_DIR%\env"
    echo 5 > "%STATE_FILE%"
    call :UpdateProgress
    goto :Stage6
)

:Stage6
if !LAST_STAGE! LEQ 6 (
    echo [%date% %time%] INFO: Stage 6 - Installing requests and tqdm >> "%LOG_FILE%"
    "!CONDA_CMD!" run -p "!INSTALL_ENV_DIR!" pip install requests tqdm
    if !ERRORLEVEL! NEQ 0 (
        echo [%date% %time%] ERROR: Failed to install requests or tqdm >> "%LOG_FILE%"
        set "LAST_STAGE=6"
        goto :error_exit
    )
    echo 6 > "%STATE_FILE%"
    call :UpdateProgress
    goto :Stage7
)

:Stage7
if !LAST_STAGE! LEQ 7 (
    echo [%date% %time%] INFO: Stage 7 - Installing Qdrant >> "%LOG_FILE%"
    echo [%date% %time%] INFO: Running install_qdrant.py from %BASE_DIR%install_qdrant.py >> "%LOG_FILE%"
    echo Starting Qdrant installation - this includes downloading and extracting up to 19GB of data, so it may take some time. Please be patient!
    echo Watch the progress bar below for Qdrant downloads - this may take a while due to a 19GB download...
    set "PYTHONUNBUFFERED=1"
    set "INSTALL_ENV_DIR=!INSTALL_DIR!\env"

    if not exist "%BASE_DIR%install_qdrant.py" (
        echo [%date% %time%] ERROR: install_qdrant.py not found at %BASE_DIR%install_qdrant.py >> "%LOG_FILE%"
        set "LAST_STAGE=7"
        goto :error_exit
    )
    "!CONDA_CMD!" run --no-capture-output -p "!INSTALL_ENV_DIR!" python "%BASE_DIR%install_qdrant.py" || (
        echo [%date% %time%] ERROR: Failed to run install_qdrant.py - check console output >> "%LOG_FILE%"
        set "LAST_STAGE=7"
        goto :error_exit
    )
    set "PYTHONUNBUFFERED="
    echo Qdrant installation files prepared successfully - now setting up the server!

    echo [%date% %time%] INFO: Starting Qdrant in its own directory: !INSTALL_DIR!\qdrant >> "%LOG_FILE%"
    echo Starting Qdrant server - this might take a moment to get ready, please wait...
    set "QDRANT_PATH=!INSTALL_DIR!\qdrant\qdrant.exe"
    start "Qdrant Server" cmd /k "cd /d !INSTALL_DIR!\qdrant && qdrant.exe"
    echo Waiting 60 seconds for Qdrant to start - hang tight, it's getting ready!
    timeout /t 60 /nobreak >nul
    echo Qdrant server is up and running - let's load the data!

    echo [%date% %time%] DEBUG: Checking Qdrant storage after startup: !INSTALL_DIR!\qdrant\storage >> "%LOG_FILE%"
    dir "!INSTALL_DIR!\qdrant\storage" >> "%LOG_FILE%" 2>&1

    echo [%date% %time%] INFO: Loading snapshot into Qdrant... >> "%LOG_FILE%"
    echo Loading 19GB snapshot into Qdrant - this could take several minutes, so grab a coffee and relax!
    set "SNAPSHOT_PATH=!INSTALL_DIR!\temp_extract\prompts_ng_gte-2103298935062809-2024-06-12-06-41-21.snapshot"
    curl -X POST "http://localhost:6333/collections/prompts_ng_gte/snapshots/upload" -F "snapshot=@!SNAPSHOT_PATH!" >> "%LOG_FILE%" 2>&1 || (
        echo [%date% %time%] ERROR: Failed to load snapshot - check Qdrant CMD window and log >> "%LOG_FILE%"
        set "LAST_STAGE=7"
        goto :error_exit
    )
    echo Snapshot loaded into Qdrant successfully - awesome, the big data's in!

    echo [%date% %time%] INFO: Verifying Qdrant collection... >> "%LOG_FILE%"
    curl "http://localhost:6333/collections/prompts_ng_gte" >> "%LOG_FILE%" 2>&1 || (
        echo [%date% %time%] ERROR: Collection not found after loading >> "%LOG_FILE%"
        set "LAST_STAGE=7"
        goto :error_exit
    )
    echo [%date% %time%] DEBUG: Checking Qdrant storage after snapshot load: !INSTALL_DIR!\qdrant\storage >> "%LOG_FILE%"
    dir "!INSTALL_DIR!\qdrant\storage" >> "%LOG_FILE%" 2>&1

    echo Qdrant setup complete - 19GB of data loaded and verified, you're crushing it!
    echo [%date% %time%] INFO: Qdrant fully installed and ready - moving to final setup! >> "%LOG_FILE%"
    echo 7 > "%STATE_FILE%"
    call :UpdateProgress
    goto :Stage8
)

:Stage8
if !LAST_STAGE! LEQ 8 (
    echo [%date% %time%] INFO: Stage 8 - Running final script >> "%LOG_FILE%"
    "!CONDA_CMD!" run --no-capture-output -p "!INSTALL_ENV_DIR!" python one_click.py
    echo 8 > "%STATE_FILE%"
    call :UpdateProgress
    set "PROGRESS=100"
    echo Progress: !PROGRESS!%% complete
    goto :end
)

:end
echo [%date% %time%] INFO: Installation completed >> "%LOG_FILE%"
del "%STATE_FILE%" 2>nul
pause
exit /b 0

:error_exit
echo [%date% %time%] ERROR: Failed at stage !LAST_STAGE! >> "%LOG_FILE%"
pause
exit /b 1