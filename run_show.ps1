# Show launcher with crash watchdog.
#   Right-click > Run with PowerShell, or:  powershell -ExecutionPolicy Bypass -File run_show.ps1
#   Extra args are passed through, e.g.:   .\run_show.ps1 --monitor 1 --panel --spout
#   Start simple (clean black canvas + one stripe layer, panel in SIMPLE mode):  .\run_show.ps1 --monitor 1 --panel --fresh
# Behaviour: ESC in the visuals (exit code 0) ends the show. Any crash restarts it within ~1 s;
# the controller restores the last live state (live_state.json), so the look resumes.
# A restart storm (5 crashes within 60 s) stops the loop so it can't spin forever.

$ErrorActionPreference = 'Continue'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here
$py = Join-Path $here '.venv\Scripts\python.exe'
$showArgs = if ($args.Count -gt 0) { $args } else { @('--monitor', '1', '--panel', '--governor', 'suggest') }
$logDir = Join-Path $here 'logs'
New-Item -ItemType Directory -Force $logDir | Out-Null
# AI worker (prompt -> textures) runs as its own low-priority process for the whole show.
# Skip it with:  $env:NO_AI = '1'
$ai = $null
if (-not $env:NO_AI) {
    $aiLog = Join-Path $logDir ("ai_{0:yyyyMMdd_HHmmss}.log" -f (Get-Date))
    $ai = Start-Process -FilePath $py -ArgumentList '-u', 'ai_worker.py', '--no-preload' -WorkingDirectory $here `
          -RedirectStandardOutput $aiLog -RedirectStandardError ($aiLog + '.err') -WindowStyle Hidden -PassThru
    Write-Host "[watchdog] AI worker started (pid $($ai.Id), log: $aiLog)"
}
$crashes = New-Object System.Collections.Generic.List[datetime]
$first = $true

while ($true) {
    $runArgs = @('-u', 'live_visuals.py') + $showArgs
    if (-not $first -and ($runArgs -contains '--panel')) {
        # don't open another browser tab on every restart; the existing panel reconnects
        $runArgs = $runArgs | Where-Object { $_ -ne '--panel' }
        $runArgs += '--remote'
    }
    if (-not $first) {
        # --fresh (clean canvas) only on the first start: a crash restart must resume what was being built
        $runArgs = $runArgs | Where-Object { $_ -ne '--fresh' }
    }
    $log = Join-Path $logDir ("show_{0:yyyyMMdd_HHmmss}.log" -f (Get-Date))
    Write-Host "[watchdog] starting: $py $($runArgs -join ' ')  (log: $log)"
    & $py @runArgs 2>&1 | Tee-Object -FilePath $log
    $code = $LASTEXITCODE
    $first = $false
    if ($code -eq 0) { Write-Host '[watchdog] clean exit (ESC) - show over.'; break }

    $now = Get-Date
    $crashes.Add($now)
    $recent = @($crashes | Where-Object { ($now - $_).TotalSeconds -lt 60 })
    Write-Host "[watchdog] crashed with code $code - restarting (crashes in last 60 s: $($recent.Count))"
    if ($recent.Count -ge 5) { Write-Host '[watchdog] too many crashes, stopping. Check the log above.'; break }
    Start-Sleep -Milliseconds 800
}

if ($ai -and -not $ai.HasExited) {
    Write-Host '[watchdog] stopping AI worker'
    Stop-Process -Id $ai.Id -Force -ErrorAction SilentlyContinue
}
