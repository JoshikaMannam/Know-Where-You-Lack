Write-Host "================================" -ForegroundColor Cyan
Write-Host "🚀 Starting Frontend Server..." -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "🔄 Starting Vite Development Server..." -ForegroundColor Yellow
Write-Host "🌐 URL: http://localhost:5173" -ForegroundColor Green
Write-Host ""

# Navigate to frontend directory and run
Set-Location frontend
npm run dev
