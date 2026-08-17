@echo off
title TERA-STRIPE - Google Drive Mount (G:)
echo ========================================================
echo   TERA-STRIPE Wildlife Platform - Google Drive Mount
echo   Mounting Google Drive as G:\ drive...
echo ========================================================
echo.
echo [INFO] Google Drive is active as G:\
echo [INFO] Keep this window minimized while working.
echo.
rclone mount gdrive: G: --vfs-cache-mode full
pause
