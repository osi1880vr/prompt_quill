cd /D "%~dp0"

set INSTALL_DIR=%cd%\installer_files
set CONDA_ROOT_PREFIX=%cd%\installer_files\conda
set INSTALL_ENV_DIR=%cd%\installer_files\env
set BASE_DIR=%cd%


if exist "%INSTALL_DIR%\bin" (
    ECHO Setting CUDA environment
    set CUDA_PATH=%INSTALL_ENV_DIR%
    set CUDA_HOME=%INSTALL_ENV_DIR%
)
@rem environment isolation
set PYTHONNOUSERSITE=1
set PYTHONPATH=
set PYTHONHOME=
set "CUDA_PATH=%INSTALL_ENV_DIR%"
set "CUDA_HOME=%CUDA_PATH%"


ECHO Startup Qdrant to upload the data
cd %INSTALL_DIR%/qdrant
start "" "%INSTALL_DIR%/qdrant/qdrant.exe"  --disable-telemetry

cd %BASE_DIR%
REM we do this to give Qdrant some time to fire up

@rem activate installer env
call "%CONDA_ROOT_PREFIX%\condabin\conda.bat" activate "%INSTALL_ENV_DIR%" || ( echo. && echo Miniconda hook not found. && goto end )

start /W "" python pq/check_qdrant_up.py

call python pq\prompt_quill_ui_qdrant.py


pause
exit