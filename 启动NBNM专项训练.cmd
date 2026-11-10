@echo off
pwsh -NoLogo -NoProfile -File "%~dp0scripts\launch_nbnm.ps1" -Mode route
if errorlevel 1 pause
