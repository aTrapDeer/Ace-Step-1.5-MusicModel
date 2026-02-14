@echo off
setlocal

set "PROMPT=%*"
if not defined PROMPT set "PROMPT=upbeat pop rap with emotional guitar"
set "PROMPT=%PROMPT:"=%"

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0test.ps1" -Prompt "%PROMPT%" -SimplePrompt -DurationSec 12 -SampleRate 44100 -Seed 42 -GuidanceScale 7.0 -Steps 50 -UseLM 1 -OutFile "test_music.wav"

if errorlevel 1 (
  echo Request failed.
  exit /b 1
)

echo Done.
endlocal
