cd /D "%~dp0"


set INSTALL_DIR=%cd%\installer_files
set CONDA_ROOT_PREFIX=%cd%\installer_files\conda
set INSTALL_ENV_DIR=%cd%\installer_files\env
set CUDA_DIR=%cd%\installer_files\env\Lib\x64
set BASE_DIR=%cd%

if exist "%INSTALL_DIR%\bin" (
    ECHO Setting CUDA environment
    set CUDA_PATH=%INSTALL_ENV_DIR%
    set CUDA_HOME=%CUDA_DIR%
)

@rem activate installer env
call "%CONDA_ROOT_PREFIX%\condabin\conda.bat" activate "%INSTALL_ENV_DIR%" || ( echo. && echo Miniconda hook not found. && goto end )

call python pg\prompt_quill_ui_qdrant.py
