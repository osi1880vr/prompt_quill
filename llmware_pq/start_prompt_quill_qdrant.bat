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


ECHO Startup Qdrant
cd %INSTALL_DIR%/qdrant
start "" "%INSTALL_DIR%/qdrant/qdrant.exe" --disable-telemetry

cd %BASE_DIR%
REM we do this to give Qdrant some time to fire up
ping 127.0.0.1 -n 6 > nul

ECHO Startup Mongo
start "" "%INSTALL_DIR%/mongo/mongodb-win32-x86_64-windows-7.0.6/bin/mongod.exe" --dbpath %INSTALL_DIR%\mongo\data


cd %BASE_DIR%
REM we do this to give Mongo some time to fire up
ping 127.0.0.1 -n 6 > nul

@rem activate installer env
call "%CONDA_ROOT_PREFIX%\condabin\conda.bat" activate "%INSTALL_ENV_DIR%" || ( echo. && echo Miniconda hook not found. && goto end )

call python pq\prompt_quill_ui_qdrant.py
