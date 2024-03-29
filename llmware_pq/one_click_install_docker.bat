@echo off
:choice
set /P c=This installer assumes you did setup qdrant as docker on localhost:6333 continue[Y/N]?
if /I "%c%" EQU "Y" goto :install
if /I "%c%" EQU "N" goto :nodocker
goto :choice

:nodocker

echo "Please install qdrant docker first"
pause
exit


:install
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
set CONDA_ROOT_PREFIX=%cd%\installer_files\conda
set INSTALL_ENV_DIR=%cd%\installer_files\env
set MINICONDA_DOWNLOAD_URL=https://repo.anaconda.com/miniconda/Miniconda3-py310_23.3.1-0-Windows-x86_64.exe
set conda_exists=F
set MONGO_TOOLS_DIR=%cd%\installer_files\mongo\mongodb-database-tools-windows-x86_64-100.9.4\bin



if not exist "%INSTALL_DIR%" (
	mkdir "%INSTALL_DIR%"
)


if exist "%INSTALL_DIR%/qdrant" (
    ECHO Startup Qdrant
    cd %INSTALL_DIR%/qdrant
    start "" "%INSTALL_DIR%/qdrant/qdrant.exe" --disable-telemetry

    cd %BASE_DIR%
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
	call "%CONDA_ROOT_PREFIX%\_conda.exe" create --no-shortcuts -y -k --prefix "%INSTALL_ENV_DIR%" python=3.11 || ( echo. && echo Conda environment creation failed. && goto end )
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


    ECHO Download llmware data
    curl -L https://civitai.com/api/download/models/371709 --output %INSTALL_DIR%/data.zip

    cd %INSTALL_DIR%

    ECHO Extract Data with unzip
    %INSTALL_DIR%/../../unzip/unzip.exe %INSTALL_DIR%/data.zip -d %INSTALL_DIR%/delete_after_setup

    ECHO Extract Qdrant Data with unzip
    %INSTALL_DIR%/../../unzip/unzip.exe %INSTALL_DIR%/delete_after_setup/prompt_quill_llmware_qdrant_v1.0.zip -d %INSTALL_DIR%/delete_after_setup


    ECHO import data to Mongo
    %MONGO_TOOLS_DIR%/mongoimport.exe --uri "mongodb://localhost:27017/llmware?retryWrites=true&w=majority" --file "installer_files/delete_after_setup/llmware.library.json" --collection "library" --jsonArray
    %MONGO_TOOLS_DIR%/mongoimport.exe --uri "mongodb://localhost:27017/llmware?retryWrites=true&w=majority" --file "installer_files/delete_after_setup/llmware.llmware_qdrant.json" --collection "llmware_qdrant" --jsonArray
    %MONGO_TOOLS_DIR%/mongoimport.exe --uri "mongodb://localhost:27017/llmware?retryWrites=true&w=majority" --file "installer_files/delete_after_setup/llmware.status.json" --collection "status" --jsonArray

    ECHO Load data into qdrant
    curl -X POST "http://localhost:6333/collections/llmware_llmwareqdrant_minilmsbert/snapshots/upload?priority=snapshot" -H "Content-Type:multipart/form-data" -H "api-key:" -F "snapshot=@%INSTALL_DIR%/delete_after_setup/llmware_llmwareqdrant_minilmsbert-1265063568362627-2024-03-03-06-52-29.snapshot"



    ECHO some cleanup
    del /f /q /a %INSTALL_DIR%\dist-qdrant.zip
    del /f /q /a %INSTALL_DIR%\qdrant-x86_64-pc-windows-msvc.zip
    del /f /q /a %INSTALL_DIR%\mongo.zip
    del /f /q /a %INSTALL_DIR%\mongo-tools.zip
    del /f /q /a %INSTALL_DIR%\data.zip
    rmdir /s /q %INSTALL_DIR%\delete_after_setup
)


call python one_click.py

