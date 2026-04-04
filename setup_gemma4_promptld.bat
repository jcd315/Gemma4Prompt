@echo off
title Gemma4 PromptLD — Auto Setup
color 0A
setlocal enabledelayedexpansion

echo.
echo ╔══════════════════════════════════════════════════════╗
echo ║       Gemma4 PromptLD — Auto Setup                                     ║
echo ║       by Brojachoeman                                                  ║
echo ╚══════════════════════════════════════════════════════╝
echo.

:: ── CONFIG ───────────────────────────────────────────────
set LLAMA_DIR=C:\llama
set MODELS_DIR=C:\models
set LLAMA_EXE=%LLAMA_DIR%\llama-server.exe
set LLAMA_ZIP=%LLAMA_DIR%\llama_install.zip
set LLAMA_URL=https://github.com/ggml-org/llama.cpp/releases/download/b8664/llama-b8664-bin-win-cuda-cu12.4-x64.zip
set GGUF_URL=https://huggingface.co/nohurry/gemma-4-26B-A4B-it-heretic-GUFF/resolve/main/gemma-4-26b-a4b-it-heretic.q4_k_m.gguf
set GGUF_FILE=%MODELS_DIR%\gemma-4-26b-a4b-it-heretic.q4_k_m.gguf
set MMPROJ_URL=https://huggingface.co/nohurry/gemma-4-26B-A4B-it-heretic-GUFF/resolve/main/gemma-4-26B-A4B-it-heretic-mmproj-bf16.gguf
set MMPROJ_FILE=%MODELS_DIR%\gemma-4-26B-A4B-it-heretic-mmproj-bf16.gguf
:: ─────────────────────────────────────────────────────────

echo [STEP 1/3] Checking llama-server...
echo.

:: Check if already in PATH
where llama-server >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ llama-server found in PATH — skipping install.
    goto :check_gguf
)

:: Check C:\llama
if exist "%LLAMA_EXE%" (
    echo ✅ llama-server found at %LLAMA_EXE% — skipping install.
    goto :check_gguf
)

:: Not found — download
echo ⚠  llama-server not found. Downloading to %LLAMA_DIR%...
echo.
echo URL: %LLAMA_URL%
echo.

if not exist "%LLAMA_DIR%" mkdir "%LLAMA_DIR%"

:: Use curl (built into Windows 10+)
curl -L --progress-bar -o "%LLAMA_ZIP%" "%LLAMA_URL%"
if %errorlevel% neq 0 (
    echo.
    echo ❌ Download failed. Check your internet connection.
    echo    You can manually download from:
    echo    %LLAMA_URL%
    echo    and extract to %LLAMA_DIR%
    pause
    exit /b 1
)

echo.
echo Extracting...
powershell -NoProfile -Command "Expand-Archive -Path '%LLAMA_ZIP%' -DestinationPath '%LLAMA_DIR%' -Force"
if %errorlevel% neq 0 (
    echo ❌ Extraction failed.
    pause
    exit /b 1
)

:: Flatten any subfolder — move everything up to C:\llama
for /d %%D in ("%LLAMA_DIR%\*") do (
    echo Flattening %%D...
    move "%%D\*" "%LLAMA_DIR%\" >nul 2>&1
    rmdir "%%D" >nul 2>&1
)

del "%LLAMA_ZIP%" >nul 2>&1

if exist "%LLAMA_EXE%" (
    echo ✅ llama-server installed at %LLAMA_EXE%
) else (
    echo ❌ llama-server.exe not found after extraction.
    echo    Check %LLAMA_DIR% manually.
    pause
    exit /b 1
)

:check_gguf
echo.
echo [STEP 2/3] Checking GGUF model...
echo.

if not exist "%MODELS_DIR%" mkdir "%MODELS_DIR%"

:: Check if any GGUF already exists
set GGUF_FOUND=0
for %%F in ("%MODELS_DIR%\*.gguf") do set GGUF_FOUND=1

if %GGUF_FOUND% == 1 (
    echo ✅ GGUF model already present in %MODELS_DIR% — skipping download.
    goto :check_python
)

echo ⚠  No GGUF found in %MODELS_DIR%.
echo.
echo Recommended model: gemma-4-26b-a4b-it-heretic Q4_K_M + mmproj (vision)
echo Size: ~15.7 GB model + 1.2 GB mmproj
echo.
set /p DOWNLOAD_GGUF="Download both now? (y/n): "
if /i "%DOWNLOAD_GGUF%" == "y" (
    echo.
    echo Downloading model GGUF — ~15.7GB, grab a brew...
    curl -L --progress-bar -o "%GGUF_FILE%" "%GGUF_URL%"
    if %errorlevel% neq 0 (
        echo ❌ GGUF download failed. Download manually and place in %MODELS_DIR%
        pause
        exit /b 1
    )
    echo ✅ Model downloaded.
    echo.
    echo Downloading mmproj — ~1.2GB (enables image input)...
    curl -L --progress-bar -o "%MMPROJ_FILE%" "%MMPROJ_URL%"
    if %errorlevel% neq 0 (
        echo ❌ mmproj download failed. You can grab it manually from HuggingFace.
        echo    Vision will not work without it but text prompting will still work.
    ) else (
        echo ✅ mmproj downloaded — vision enabled.
    )
) else (
    echo.
    echo Skipped. Place your GGUF in: %MODELS_DIR%
    echo Then restart ComfyUI — it will appear in the node dropdown.
)

:check_python
echo.
echo [STEP 3/3] Checking Python dependencies...
echo.

pip show requests >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing requests...
    pip install requests
) else (
    echo ✅ requests already installed.
)

:: Done
echo.
echo ╔══════════════════════════════════════════════════════╗
echo ║   ✅ Setup Complete!                                 ║
echo ╠══════════════════════════════════════════════════════╣
echo ║                                                      ║
echo ║   llama-server : %LLAMA_EXE%
echo ║   Models folder: %MODELS_DIR%                       ║
echo ║   Model        : gemma-4-26b-a4b-it-heretic Q4_K_M  ║
echo ║   Vision       : mmproj-bf16 (image input enabled)  ║
echo ║                                                      ║
echo ║   Next steps:                                        ║
echo ║   1. Restart ComfyUI                                 ║
echo ║   2. Add the Gemma4 Prompt Engineer node             ║
echo ║   3. Hit PREVIEW — node handles the rest             ║
echo ║                                                      ║
echo ╚══════════════════════════════════════════════════════╝
echo.
pause
