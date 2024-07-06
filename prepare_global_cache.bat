@echo off

cd /D "%~dp0"

set PATH=%PATH%;%SystemRoot%\system32

@rem config
set BASE_DIR=%cd%
set CACHE_DIR=%cd%\installer_cache


if not exist "%CACHE_DIR%" (
	mkdir "%CACHE_DIR%"
)

if not exist "%CACHE_DIR%/llama_index" (
	mkdir "%CACHE_DIR%/llama_index"
)



if not exist "%CACHE_DIR%/qdrant-x86_64-pc-windows-msvc.zip" (
    ECHO Download Qdrant Portable Version
    curl -L  https://github.com/qdrant/qdrant/releases/download/v1.10.0/qdrant-x86_64-pc-windows-msvc.zip --output %CACHE_DIR%/qdrant-x86_64-pc-windows-msvc.zip
) else (
    ECHO skipped Download Qdrant Portable Version
)


if not exist "%CACHE_DIR%/dist-qdrant.zip" (
    ECHO Download Qdrant Web UI
    curl -L https://github.com/qdrant/qdrant-web-ui/releases/download/v0.1.29/dist-qdrant.zip --output %CACHE_DIR%/dist-qdrant.zip
) else (
    ECHO skipped Download Qdrant Web UI
)

if not exist "%CACHE_DIR%/data.zip" (
    ECHO Download LLama-index QDrant data
    curl -L https://civitai.com/api/download/models/567736 --output %CACHE_DIR%/data.zip
) else (
    ECHO skipped Download LLama-index QDrant data
)







:end
exit