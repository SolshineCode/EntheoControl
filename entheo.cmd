@echo off
REM EntheoControl launcher (public). Usage:
REM   entheo              start the show on monitor 1 (projector) with the phone/laptop control panel
REM   entheo 2            projector is monitor 2
REM   entheo stop         stop the show
REM Extra args are passed to run_show.ps1 (e.g. entheo 1 --fresh).
setlocal
if /I "%~1"=="stop" (
  powershell -NoProfile -Command "Get-CimInstance Win32_Process | ? { $_.CommandLine -match 'run_show.ps1|live_visuals.py|ai_worker.py|camera_worker.py' -and $_.ProcessId -ne $PID } | %% { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
  echo EntheoControl stopped.
  exit /b 0
)
set MON=%~1
if "%MON%"=="" set MON=1
shift
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_show.ps1" --monitor %MON% --panel --remote --governor suggest %1 %2 %3 %4 %5
