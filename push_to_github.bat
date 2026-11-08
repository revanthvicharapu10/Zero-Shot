@echo off
title Push RareVision AI to GitHub
echo Pushing RareVision AI to revanthvicharapu10/Zero-Shot...
git push -u origin main
if %errorlevel% neq 0 (
    echo.
    echo [!] 403 Permission Denied: Your Git account 'kotasrilakshmi066-cpu' needs collaborator access on revanthvicharapu10/Zero-Shot.
)
pause
