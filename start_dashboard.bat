@echo off
title TERA-STRIPE - Tactical Wildlife Dashboard
echo ========================================================
echo   TERA-STRIPE Wildlife Intelligence Platform Dashboard
echo   Starting web server on http://localhost:8501...
echo ========================================================
echo.
cd /d "c:\all projects\nagpurhack"
start http://localhost:8501
python dashboard/app.py
pause
