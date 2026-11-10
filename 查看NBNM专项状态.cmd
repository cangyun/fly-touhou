@echo off
pwsh -NoLogo -NoProfile -File "%~dp0scripts\show_nbnm.ps1" -Watch
if errorlevel 1 pause
