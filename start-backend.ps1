# Set Environment Variables
$env:DB_PASSWORD="YOUR_DB_PASSWORD_HERE"
$env:DB_USERNAME="root"
$env:JWT_SECRET="YourVerySecureSecretKeyThatIsAtLeast32CharactersLong123456"
$env:GROQ_API_KEY="YOUR_GROQ_API_KEY_HERE"

Write-Host "================================" -ForegroundColor Cyan
Write-Host "🚀 Starting Backend Server..." -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "✅ Environment Variables Set:" -ForegroundColor Green
Write-Host "   DB_PASSWORD: ********"
Write-Host "   DB_USERNAME: $env:DB_USERNAME"
Write-Host "   JWT_SECRET: ********"
Write-Host "   GROQ_API_KEY: ********"
Write-Host ""
Write-Host "🔄 Starting Spring Boot Application..." -ForegroundColor Yellow
Write-Host ""

# Navigate to backend directory and run
Set-Location java-backend
mvn spring-boot:run
