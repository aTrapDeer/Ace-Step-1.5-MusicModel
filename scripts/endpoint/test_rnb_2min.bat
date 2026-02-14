@echo off
setlocal

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0test.ps1" -RnbPopRap2Min -DurationSec 120 -SampleRate 44100 -Seed 42 -GuidanceScale 7.0 -Steps 8 -UseLM 1 -OutFile "test_rnb_pop_rap_2min.wav"

if errorlevel 1 (
  echo Request failed.
  exit /b 1
)

echo Done.
endlocal
