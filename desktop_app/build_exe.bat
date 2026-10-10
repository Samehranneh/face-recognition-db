@echo off

echo.
echo COMPILING YOUR APP TO .EXE...
echo (This may take a few minutes. Please wait.)
echo.

REM --- Run PyInstaller and redirect ALL output (both standard and error) to a log file ---
REM Ensure your biometric_registry.py file is in the same directory as this batch file!
python -m PyInstaller --onefile biometric_registry.py > build_log.txt 2>&1

REM Check the exit code of the last command
if errorlevel 1 (
echo.
echo --------------------------------------------------------------------------------
echo !!! COMPILATION FAILED !!!
echo The PyInstaller command returned an error.
echo Please check the 'build_log.txt' file in this folder for details on the error.
echo --------------------------------------------------------------------------------
) else (
echo.
echo DONE!
echo.
echo Your new application is located in the "dist" folder.
)

echo.
echo Press any key to continue.

pause > nul