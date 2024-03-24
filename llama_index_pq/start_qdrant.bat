set INSTALL_DIR=%cd%\installer_files
set CONDA_ROOT_PREFIX=%cd%\installer_files\conda
set INSTALL_ENV_DIR=%cd%\installer_files\env
set BASE_DIR=%cd%

if exist "%INSTALL_DIR%\bin" (
    ECHO Setting CUDA environment
    set CUDA_PATH=%INSTALL_ENV_DIR%
    set CUDA_HOME=%INSTALL_ENV_DIR%
)

ECHO Startup Qdrant to upload the data
cd %INSTALL_DIR%/qdrant
start "" "%INSTALL_DIR%/qdrant/qdrant.exe"

cd %BASE_DIR%
REM we do this to give Qdrant some time to fire up
ping 127.0.0.1 -n 6 > nul


@rem activate installer env
call "%CONDA_ROOT_PREFIX%\condabin\conda.bat" activate "%INSTALL_ENV_DIR%" || ( echo. && echo Miniconda hook not found. && goto end )

call python prompt_quill_ui_qdrant.py
