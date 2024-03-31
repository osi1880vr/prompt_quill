@echo off

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

@rem config
set BASE_DIR=%cd%
set CACHE_DIR=%cd%\installer_cache


if not exist "%CACHE_DIR%" (
	mkdir "%CACHE_DIR%"
)

if not exist "%CACHE_DIR%/qdrant-x86_64-pc-windows-msvc.zip" (
    ECHO Download Qdrant Portable Version
    curl -L https://github.com/qdrant/qdrant/releases/download/v1.8.1/qdrant-x86_64-pc-windows-msvc.zip --output %CACHE_DIR%/qdrant-x86_64-pc-windows-msvc.zip
)

if not exist "%CACHE_DIR%/dist-qdrant.zip" (
    ECHO Download Qdrant Web UI
    curl -L https://github.com/qdrant/qdrant-web-ui/releases/download/v0.1.22/dist-qdrant.zip --output %CACHE_DIR%/dist-qdrant.zip
)

if not exist "%CACHE_DIR%/data.zip" (
    ECHO Download LLama-index QDrant data
    curl -L https://civitai.com/api/download/models/407093 --output %CACHE_DIR%/data.zip
)



:end
exit