@echo off
rem Builds the native search core. Run from a "x64 Native Tools Command Prompt for VS".
cd /d "%~dp0"
cl /nologo /O2 /EHsc /LD /Fe:chess_engine\native\deeptactics_native_runtime_v2.dll /Fo:chess_engine\native\ chess_engine\native\native_eval.cpp
if errorlevel 1 (echo Build failed. & exit /b 1)
del /q chess_engine\native\*.obj chess_engine\native\*.lib chess_engine\native\*.exp 2>nul
echo Built chess_engine\native\deeptactics_native_runtime_v2.dll
