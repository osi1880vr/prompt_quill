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

if not exist "%CACHE_DIR%/llmware" (
	mkdir "%CACHE_DIR%/llmware"
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

if not exist "%CACHE_DIR%/llmware/data.zip" (
    ECHO Download llmware QDrant data
    curl -L https://civitai.com/api/download/models/420489 --output %CACHE_DIR%/llmware/data.zip
) else (
    ECHO skipped Download llmware QDrant data, already exists
)


if not exist "%CACHE_DIR%/llama_index/data.zip" (
    ECHO Download LLama-index QDrant data
    curl -L https://civitai.com/api/download/models/407093 --output %CACHE_DIR%/llama_index/data.zip
) else (
    ECHO skipped Download LLama-index QDrant data
)








:end
exit