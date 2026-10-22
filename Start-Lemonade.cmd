@echo off
setlocal
title ggml-xdna for Lemonade
powershell.exe -NoLogo -NoProfile -STA -ExecutionPolicy Bypass -File "%~dp0tools\start-lemonade.ps1" -Interactive %*
set "lemonade_exit=%ERRORLEVEL%"
if not "%lemonade_exit%"=="0" (
    echo.
    echo Lemonade did not finish normally. The message above explains why.
    pause
)
exit /b %lemonade_exit%
