@echo off

REM Create folders from p0 to p10
for /l %%i in (0,1,10) do (
    mkdir "p%%i" 2>nul
    echo Created folder p%%i
)

echo All folders created successfully!
pause 