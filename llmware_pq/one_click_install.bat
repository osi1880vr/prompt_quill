rem @echo off

:: If you don't already have Git, download Git-SCM and install it here: https://git-scm.com/download/win
WHERE git >nul 2>nul
IF %ERRORLEVEL% NEQ 0 (
	ECHO:
	ECHO You will need to install git first before running this script. Please download it at https://git-scm.com/download/win
	ECHO:
	pause
	exit
)

cd /D "%~dp0"

set PATH=%PATH%;%SystemRoot%\system32

echo "%CD%"| findstr /C:" " >nul && echo This script relies on Miniconda which can not be silently installed under a path with spaces. && goto end

@rem Check for special characters in installation path
set "SPCHARMESSAGE="WARNING: Special characters were detected in the installation path!" "         This can cause the installation to fail!""
echo "%CD%"| findstr /R /C:"[!#\$%&()\*+,;<=>?@\[\]\^`{|}~]" >nul && (
	call :PrintBigMessage %SPCHARMESSAGE%
)
set SPCHARMESSAGE=

@rem fix failed install when installing to a separate drive
set TMP=%cd%\installer_files
set TEMP=%cd%\installer_files

@rem deactivate existing conda envs as needed to avoid conflicts
(call conda deactivate && call conda deactivate && call conda deactivate) 2>nul

@rem config
set BASE_DIR=%cd%
set INSTALL_DIR=%cd%\installer_files
set CACHE_DIR=%cd%\installer_cache
set CONDA_ROOT_PREFIX=%cd%\installer_files\conda
set INSTALL_ENV_DIR=%cd%\installer_files\env
set MINICONDA_DOWNLOAD_URL=https://repo.anaconda.com/miniconda/Miniconda3-py310_23.3.1-0-Windows-x86_64.exe
set conda_exists=F
set MONGO_TOOLS_DIR=%cd%\installer_files\mongo\mongodb-database-tools-windows-x86_64-100.9.4\bin



if not exist "%INSTALL_DIR%" (
	mkdir "%INSTALL_DIR%"
)
if not exist "%INSTALL_DIR%/delete_after_setup" (
	mkdir "%INSTALL_DIR%/delete_after_setup"
)

if exist "%INSTALL_DIR%/qdrant" (
    ECHO Startup Qdrant
    cd %INSTALL_DIR%/qdrant
    start "" "%INSTALL_DIR%/qdrant/qdrant.exe" --disable-telemetry

    cd %BASE_DIR%
    ping 127.0.0.1 -n 6 > nul

)


if exist "%INSTALL_DIR%/mongo" (
    ECHO Startup Mongo
    start "" "%INSTALL_DIR%/mongo/mongodb-win32-x86_64-windows-7.0.6/bin/mongod.exe" --dbpath %INSTALL_DIR%\mongo\data

    cd %BASE_DIR%
    REM we do this to give Mongo some time to fire up
    ping 127.0.0.1 -n 6 > nul
)


@rem figure out whether git and conda needs to be installed
call "%CONDA_ROOT_PREFIX%\_conda.exe" --version >nul 2>&1
if "%ERRORLEVEL%" EQU "0" set conda_exists=T

@rem (if necessary) install git and conda into a contained environment
@rem download conda
if "%conda_exists%" == "F" (
	echo Downloading Miniconda from %MINICONDA_DOWNLOAD_URL% to %INSTALL_DIR%\miniconda_installer.exe

	call curl -Lk "%MINICONDA_DOWNLOAD_URL%" > "%INSTALL_DIR%\miniconda_installer.exe" || ( echo. && echo Miniconda failed to download. && goto end )

	echo Installing Miniconda to %CONDA_ROOT_PREFIX%
	start /wait "" "%INSTALL_DIR%\miniconda_installer.exe" /InstallationType=JustMe /NoShortcuts=1 /AddToPath=0 /RegisterPython=0 /NoRegistry=1 /S /D=%CONDA_ROOT_PREFIX%

	@rem test the conda binary
	echo Miniconda version:
	call "%CONDA_ROOT_PREFIX%\_conda.exe" --version || ( echo. && echo Miniconda not found. && goto end )
)

@rem create the installer env
if not exist "%INSTALL_ENV_DIR%" (
	echo Packages to install: %PACKAGES_TO_INSTALL%
	call "%CONDA_ROOT_PREFIX%\_conda.exe" create --no-shortcuts -y -k --prefix "%INSTALL_ENV_DIR%" python=3.10 || ( echo. && echo Conda environment creation failed. && goto end )
)

@rem check if conda environment was actually created
if not exist "%INSTALL_ENV_DIR%\python.exe" ( echo. && echo Conda environment is empty. && goto end )


@rem environment isolation
set PYTHONNOUSERSITE=1
set PYTHONPATH=
set PYTHONHOME=
set "CUDA_PATH=%INSTALL_ENV_DIR%"
set "CUDA_HOME=%INSTALL_ENV_DIR%"

@rem activate installer env
call "%CONDA_ROOT_PREFIX%\condabin\conda.bat" activate "%INSTALL_ENV_DIR%" || ( echo. && echo Miniconda hook not found. && goto end )

ECHO cleanup miniconda installer
del /f %INSTALL_DIR%\miniconda_installer.exe




if not exist "%INSTALL_DIR%/qdrant" (


    if not exist "%CACHE_DIR%/qdrant-x86_64-pc-windows-msvc.zip" (
        ECHO Download Qdrant Portable Version
        curl -L https://github.com/qdrant/qdrant/releases/download/v1.8.1/qdrant-x86_64-pc-windows-msvc.zip --output %INSTALL_DIR%/qdrant-x86_64-pc-windows-msvc.zip
    ) else (
        xcopy %CACHE_DIR%\qdrant-x86_64-pc-windows-msvc.zip %INSTALL_DIR%
    )

    if not exist "%CACHE_DIR%/dist-qdrant.zip" (
        ECHO Download Qdrant Web UI
        curl -L https://github.com/qdrant/qdrant-web-ui/releases/download/v0.1.22/dist-qdrant.zip --output %INSTALL_DIR%/dist-qdrant.zip
    ) else (
        xcopy %CACHE_DIR%\dist-qdrant.zip %INSTALL_DIR%
    )

    if not exist "%CACHE_DIR%/mongo.zip" (
        ECHO Download Mongo DB
        curl -L https://fastdl.mongodb.org/windows/mongodb-windows-x86_64-7.0.6.zip --output %INSTALL_DIR%/mongo.zip
    ) else (
        xcopy %CACHE_DIR%\mongo.zip %INSTALL_DIR%
    )

    if not exist "%CACHE_DIR%/mongo-tools.zip" (
        ECHO Download Mongo DB Tools
        curl -L https://fastdl.mongodb.org/tools/db/mongodb-database-tools-windows-x86_64-100.9.4.zip --output %INSTALL_DIR%/mongo-tools.zip
    ) else (
        xcopy %CACHE_DIR%\mongo-tools.zip %INSTALL_DIR%
    )

    if not exist "%CACHE_DIR%/data.zip" (
        ECHO Download llmware QDrant data
        curl -L https://civitai.com/api/download/models/420489 --output %INSTALL_DIR%/data.zip
    ) else (
        xcopy %CACHE_DIR%\data.zip %INSTALL_DIR%
    )


    ECHO Extract Qdrant with unzip
    %INSTALL_DIR%/../../unzip/unzip.exe %INSTALL_DIR%/qdrant-x86_64-pc-windows-msvc.zip -d %INSTALL_DIR%/qdrant

    ECHO Extract Qdrant web UI with unzip
    %INSTALL_DIR%/../../unzip/unzip.exe %INSTALL_DIR%/dist-qdrant.zip -d %INSTALL_DIR%/qdrant

    ECHO Extract Mongo DB with unzip
    %INSTALL_DIR%/../../unzip/unzip.exe %INSTALL_DIR%/mongo.zip -d %INSTALL_DIR%/mongo

    ECHO Extract Mongo Tools with unzip
    %INSTALL_DIR%/../../unzip/unzip.exe %INSTALL_DIR%/mongo-tools.zip -d %INSTALL_DIR%/mongo

    ECHO rename the dist folder to static
    cd %INSTALL_DIR%/qdrant
    ren "dist" "static"

    cd %INSTALL_DIR%

    ECHO Extract Data with unzip
    %INSTALL_DIR%/../../unzip/unzip.exe %INSTALL_DIR%/data.zip -d %INSTALL_DIR%/delete_after_setup

    ECHO Extract Qdrant Data with unzip
    %INSTALL_DIR%/../../unzip/unzip.exe %INSTALL_DIR%/delete_after_setup/prompt_quill_llmware_qdrant_meta_v1.0.zip -d %INSTALL_DIR%/delete_after_setup

    ECHO Extract Mongo with unzip
    %INSTALL_DIR%/../../unzip/unzip.exe %INSTALL_DIR%/delete_after_setup/mongo_data.zip -d %INSTALL_DIR%/delete_after_setup

    if not exist "%INSTALL_DIR%/mongo/data" (
        mkdir "%INSTALL_DIR%/mongo/data"
    )

    ECHO Startup Qdrant to upload the data
    cd %INSTALL_DIR%/qdrant
    start "" "%INSTALL_DIR%/qdrant/qdrant.exe"

    REM we do this to give Qdrant some time to fire up
    ping 127.0.0.1 -n 6 > nul


    ECHO Startup Mongo to upload the data
    start "" "%INSTALL_DIR%/mongo/mongodb-win32-x86_64-windows-7.0.6/bin/mongod.exe" --dbpath %INSTALL_DIR%\mongo\data


    cd %BASE_DIR%
    REM we do this to give Mongo some time to fire up
    ping 127.0.0.1 -n 6 > nul


    ECHO import data to Mongo
    %MONGO_TOOLS_DIR%/mongoimport.exe --uri "mongodb://localhost:27017/llmware?retryWrites=true&w=majority" --file "installer_files/delete_after_setup/mongo_data/llmware.library.json" --collection "library" --jsonArray
    %MONGO_TOOLS_DIR%/mongoimport.exe --uri "mongodb://localhost:27017/llmware?retryWrites=true&w=majority" --file "installer_files/delete_after_setup/mongo_data/llmware.llmware_meta_qdrant.json" --collection "llmware_meta_qdrant" --jsonArray
    %MONGO_TOOLS_DIR%/mongoimport.exe --uri "mongodb://localhost:27017/llmware?retryWrites=true&w=majority" --file "installer_files/delete_after_setup/mongo_data/llmware.status.json" --collection "status" --jsonArray

    ECHO Load data into qdrant
    curl -X POST "http://localhost:6333/collections/llmware_llmwareqdrant_minilmsbert/snapshots/upload?priority=snapshot" -H "Content-Type:multipart/form-data" -H "api-key:" -F "snapshot=@%INSTALL_DIR%/delete_after_setup/llmware_llmwaremetaqdrant_minilmsbert-3474994170629559-2024-03-31-07-00-42.snapshot"



    ECHO some cleanup
    del /f /q /a %INSTALL_DIR%\dist-qdrant.zip
    del /f /q /a %INSTALL_DIR%\qdrant-x86_64-pc-windows-msvc.zip
    del /f /q /a %INSTALL_DIR%\mongo.zip
    del /f /q /a %INSTALL_DIR%\mongo-tools.zip
    del /f /q /a %INSTALL_DIR%\data.zip
    rmdir /s /q %INSTALL_DIR%\delete_after_setup
)


call python one_click.py

:PrintBigMessage
echo. && echo.
echo *******************************************************************
for %%M in (%*) do echo * %%~M
echo *******************************************************************
echo. && echo.
exit /b