# Open Deployment Guide in Browser
param(
    [switch]$OpenGuide,
    [switch]$ShowKey,
    [switch]$OpenSites
)

if ($OpenGuide) {
    Write-Host "Opening deployment guide..." -ForegroundColor Cyan
    Start-Process "RENDER_DEPLOYMENT_ACTION_PLAN.md"
}

if ($ShowKey) {
    Write-Host "`nYour GROQ API Key:" -ForegroundColor Yellow
    cat .env | Select-String GROQ_API_KEY
}

if ($OpenSites) {
    Write-Host "Opening deployment sites..." -ForegroundColor Cyan
    Start-Process "https://aiven.io/"
    Start-Sleep -Seconds 1
    Start-Process "https://render.com/"
}

if (-not $OpenGuide -and -not $ShowKey -and -not $OpenSites) {
    Write-Host ""
    Write-Host "==================================================================" -ForegroundColor Cyan
    Write-Host "          KNOW WHERE YOU LACK - DEPLOYMENT HELPER" -ForegroundColor Green
    Write-Host "==================================================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Usage:" -ForegroundColor Yellow
    Write-Host "  .\start-deployment.ps1 -OpenGuide    # Open step-by-step guide" -ForegroundColor White
    Write-Host "  .\start-deployment.ps1 -ShowKey      # Show your GROQ API key" -ForegroundColor White
    Write-Host "  .\start-deployment.ps1 -OpenSites    # Open Aiven & Render" -ForegroundColor White
    Write-Host ""
    Write-Host "Quick Deploy:" -ForegroundColor Yellow
    Write-Host "  1. Run: .\start-deployment.ps1 -OpenSites" -ForegroundColor Cyan
    Write-Host "  2. Follow: RENDER_DEPLOYMENT_ACTION_PLAN.md" -ForegroundColor Cyan
    Write-Host "  3. Deploy in ~20 minutes!" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Your GROQ API Key:" -ForegroundColor Yellow
    cat .env | Select-String GROQ_API_KEY
    Write-Host ""
    Write-Host "==================================================================" -ForegroundColor Cyan
    Write-Host ""
}
