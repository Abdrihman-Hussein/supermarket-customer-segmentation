@echo off
title Supermarket Segmentation - Next.js Dashboard
cd /d "C:\Users\hp\Downloads\kuas\nextjs"

:start
cls
echo ============================================
echo Supermarket Customer Segmentation Dashboard
echo ============================================
echo.
echo [1] Start on port 3000 (default)
echo [2] Start on port 3001 (recommended if 3000 is busy)
echo [3] Start on port 3002
echo [4] Check if port is free
echo [0] Exit
echo.
set /p choice="Choose option [1-4,0]: "

if "%choice%"=="1" (
    echo Starting Next.js on port 3000...
    next dev
    goto end
)
if "%choice%"=="2" (
    echo Starting Next.js on port 3001...
    next dev -p 3001
    goto end
)
if "%choice%"=="3" (
    echo Starting Next.js on port 3002...
    next dev -p 3002
    goto end
)
if "%choice%"=="4" (
    echo Checking port availability...
    powershell -command "Test-NetConnection -Port 3000 -InformationLevel Quiet"
    goto start
)
if "%choice%"=="0" (
    echo Exiting...
    goto end
)
echo Invalid option. Try again.
goto start

:echo
echo.
echo ============================================
echo Installation Check
echo ============================================
echo.
echo [1] Install dependencies (npm install)
echo [2] Rebuild node modules
echo [3] Return to main menu
echo.
set /p choice2="Choose option [1-3]: "

if "%choice2%"=="1" (
    echo Installing npm dependencies...
    npm install --legacy-peer-deps
    goto start
)
if "%choice2%"=="2" (
    echo Removing and reinstalling node modules...
    rmdir /s /q node_modules
    rm package-lock.json
    npm install --legacy-peer-deps
    goto start
)
if "%choice2%"=="3" (
    goto start
)

:end
echo.
echo ============================================
echo Done.
echo If port 3000/3001/3002 is busy, try another port.
echo Open http://localhost:3000 or http://localhost:3001 in browser.
echo ============================================
timeout /t 5 /nobreak >nul