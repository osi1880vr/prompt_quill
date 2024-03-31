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
) else (
    ECHO skipped Download Qdrant Portable Version, already exists
)


if not exist "%CACHE_DIR%/dist-qdrant.zip" (
    ECHO Download Qdrant Web UI
    curl -L https://github.com/qdrant/qdrant-web-ui/releases/download/v0.1.22/dist-qdrant.zip --output %CACHE_DIR%/dist-qdrant.zip
) else (
    ECHO skipped Download Qdrant Web UI, already exists
)

if not exist "%CACHE_DIR%/mongo.zip" (
    ECHO Download Mongo DB
    curl -L https://fastdl.mongodb.org/windows/mongodb-windows-x86_64-7.0.6.zip --output %CACHE_DIR%/mongo.zip
) else (
    ECHO skipped Download Mongo DB, already exists
)

if not exist "%CACHE_DIR%/mongo-tools.zip" (
    ECHO Download Mongo DB Tools
    curl -L https://fastdl.mongodb.org/tools/db/mongodb-database-tools-windows-x86_64-100.9.4.zip --output %CACHE_DIR%/mongo-tools.zip
) else (
    ECHO skipped Mongo DB Tools, already exists
)

if not exist "%CACHE_DIR%/data.zip" (
    ECHO Download llmware QDrant data
    curl -L https://civitai.com/api/download/models/420489 --output %CACHE_DIR%/data.zip
) else (
    ECHO skipped Download llmware QDrant data, already exists
)










:end
exit