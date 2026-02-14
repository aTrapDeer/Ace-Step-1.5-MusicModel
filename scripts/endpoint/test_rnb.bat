@echo off
setlocal

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0test.ps1" -RnbLoveTemptation -DurationSec 24 -SampleRate 44100 -Seed 42 -GuidanceScale 7.0 -Steps 8 -UseLM 1 -OutFile "test_rnb_music.wav"

if errorlevel 1 (
  echo Request failed.
  exit /b 1
)

echo Done.
endlocal
