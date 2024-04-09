@echo off

cd /D "%~dp0"

set PATH=%PATH%;%SystemRoot%\system32

@rem config
set BASE_DIR=%cd%
set CACHE_DIR=%cd%\installer_cache
set LLAMA_CACHE_DIR=%cd%\llama_index_pq\installer_cache
set LLMWARE_CACHE_DIR=%cd%\llmware_pq\installer_cache

if not exist "%LLAMA_CACHE_DIR%" (
	mkdir "%LLAMA_CACHE_DIR%"
)
if not exist "%LLMWARE_CACHE_DIR%" (
	mkdir "%LLMWARE_CACHE_DIR%"
)


xcopy %CACHE_DIR%\qdrant-x86_64-pc-windows-msvc.zip %LLMWARE_CACHE_DIR%
xcopy %CACHE_DIR%\dist-qdrant.zip %LLMWARE_CACHE_DIR%
xcopy %CACHE_DIR%\mongo.zip %LLMWARE_CACHE_DIR%
xcopy %CACHE_DIR%\mongo-tools.zip %LLMWARE_CACHE_DIR%
xcopy %CACHE_DIR%\llmware\data.zip %LLMWARE_CACHE_DIR%


xcopy %CACHE_DIR%\qdrant-x86_64-pc-windows-msvc.zip %LLAMA_CACHE_DIR%
xcopy %CACHE_DIR%\dist-qdrant.zip %LLAMA_CACHE_DIR%
xcopy %CACHE_DIR%\llama_index\data.zip %LLAMA_CACHE_DIR%



:end
exit