# parkrun Insights — demo startup script
# Starts the API, worker, and a Cloudflare tunnel, then tells you what to paste into Vercel.
#
# Requirements (one-time installs):
#   winget install cloudflare.cloudflared

Set-StrictMode -Off
$ErrorActionPreference = "Continue"

# ── 1. Ensure cloudflared is available (auto-download if missing) ─────────────
$cloudflaredCmd = "cloudflared"
$localBin = Join-Path $PSScriptRoot "cloudflared.exe"

if (-not (Get-Command cloudflared -ErrorAction SilentlyContinue)) {
    if (Test-Path $localBin) {
        $cloudflaredCmd = $localBin
        Write-Host "Using local cloudflared.exe" -ForegroundColor DarkGray
    } else {
        Write-Host "cloudflared not found — downloading it now..." -ForegroundColor Yellow
        $url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
        try {
            Invoke-WebRequest -Uri $url -OutFile $localBin -UseBasicParsing
            $cloudflaredCmd = $localBin
            Write-Host "  Downloaded to $localBin" -ForegroundColor Green
        } catch {
            Write-Host "  Download failed: $_" -ForegroundColor Red
            Write-Host "  Download manually from: $url" -ForegroundColor Yellow
            exit 1
        }
    }
}

# ── 2. Kill anything already on port 8000 ────────────────────────────────────
$existing = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "Killing existing process on port 8000 (PID $($existing.OwningProcess))..." -ForegroundColor Yellow
    Stop-Process -Id $existing.OwningProcess -Force -ErrorAction SilentlyContinue
    Start-Sleep -Milliseconds 500
}

# ── 3. Locate the backend directory ──────────────────────────────────────────
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$backend = Join-Path $root "backend"

# ── 4. Start the API ─────────────────────────────────────────────────────────
Write-Host ""
Write-Host "Starting API server..." -ForegroundColor Green
$api = Start-Process -PassThru -NoNewWindow `
    -FilePath "python" `
    -ArgumentList "-m", "uvicorn", "app.main:app", "--port", "8000" `
    -WorkingDirectory $backend `
    -RedirectStandardOutput "$env:TEMP\parkrun_api.log" `
    -RedirectStandardError  "$env:TEMP\parkrun_api_err.log"
Write-Host "  API PID: $($api.Id)" -ForegroundColor DarkGray

# Wait for API to be ready
$ready = $false
for ($i = 0; $i -lt 20; $i++) {
    Start-Sleep -Milliseconds 500
    try {
        $r = Invoke-WebRequest -Uri "http://localhost:8000/api/health" -UseBasicParsing -TimeoutSec 1 -ErrorAction Stop
        if ($r.StatusCode -eq 200) { $ready = $true; break }
    } catch {}
}
if (-not $ready) {
    Write-Host "  API did not start in time. Check $env:TEMP\parkrun_api_err.log" -ForegroundColor Red
    exit 1
}
Write-Host "  API is up." -ForegroundColor Green

# ── 5. Start the worker ───────────────────────────────────────────────────────
Write-Host "Starting pipeline worker..." -ForegroundColor Green
$worker = Start-Process -PassThru -NoNewWindow `
    -FilePath "python" `
    -ArgumentList "-m", "app.worker.pipeline" `
    -WorkingDirectory $backend `
    -RedirectStandardOutput "$env:TEMP\parkrun_worker.log" `
    -RedirectStandardError  "$env:TEMP\parkrun_worker_err.log"
Write-Host "  Worker PID: $($worker.Id)" -ForegroundColor DarkGray

# ── 6. Start Cloudflare tunnel and capture URL ────────────────────────────────
Write-Host "Starting Cloudflare tunnel..." -ForegroundColor Green
$tunnelLog = "$env:TEMP\parkrun_tunnel.log"
Remove-Item $tunnelLog -ErrorAction SilentlyContinue

$tunnel = Start-Process -PassThru -NoNewWindow `
    -FilePath $cloudflaredCmd `
    -ArgumentList "tunnel", "--url", "http://localhost:8000", "--no-autoupdate" `
    -RedirectStandardOutput $tunnelLog `
    -RedirectStandardError  $tunnelLog

$tunnelUrl = $null
for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Milliseconds 500
    if (Test-Path $tunnelLog) {
        $content = Get-Content $tunnelLog -Raw -ErrorAction SilentlyContinue
        if ($content -match 'https://[a-z0-9\-]+\.trycloudflare\.com') {
            $tunnelUrl = $Matches[0]
            break
        }
    }
}

if (-not $tunnelUrl) {
    Write-Host "  Could not extract tunnel URL from cloudflared output." -ForegroundColor Red
    Write-Host "  Check $tunnelLog for details." -ForegroundColor Yellow
    exit 1
}

# ── 7. Print what to do next ──────────────────────────────────────────────────
Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host " Demo is running!" -ForegroundColor Green
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host ""
Write-Host " Tunnel URL (backend):  $tunnelUrl" -ForegroundColor Yellow
Write-Host ""
Write-Host " ACTION REQUIRED — update Vercel before sharing the demo URL:" -ForegroundColor White
Write-Host "  1. Go to: vercel.com → your project → Settings → Environment Variables"
Write-Host "  2. Set NEXT_PUBLIC_API_BASE_URL = $tunnelUrl"
Write-Host "  3. Save, then click Redeploy (takes ~30 seconds)"
Write-Host ""
Write-Host " Logs:"
Write-Host "  API    → $env:TEMP\parkrun_api_err.log"
Write-Host "  Worker → $env:TEMP\parkrun_worker_err.log"
Write-Host "  Tunnel → $tunnelLog"
Write-Host ""
Write-Host " Press Ctrl+C to stop everything." -ForegroundColor DarkGray
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host ""

# Keep the script alive so Ctrl+C stops gracefully
try {
    while ($true) { Start-Sleep -Seconds 30 }
} finally {
    Write-Host "Stopping..." -ForegroundColor Yellow
    Stop-Process -Id $api.Id    -Force -ErrorAction SilentlyContinue
    Stop-Process -Id $worker.Id -Force -ErrorAction SilentlyContinue
    Stop-Process -Id $tunnel.Id -Force -ErrorAction SilentlyContinue
    Write-Host "All processes stopped." -ForegroundColor Green
}
