# Test Backend API
Write-Host "`n=== Testing Backend API ===" -ForegroundColor Cyan

# Test 1: Register a new user
Write-Host "`n1. Testing Registration..." -ForegroundColor Yellow
$registerBody = @{
    username = "testuser"
    fullName = "Test User"
    email = "test@example.com"
    password = "test1234"
} | ConvertTo-Json

try {
    $registerResponse = Invoke-RestMethod -Uri "http://localhost:8082/api/auth/register" -Method POST -ContentType "application/json" -Body $registerBody
    Write-Host "Success: Registration successful!" -ForegroundColor Green
    Write-Host "  User ID: $($registerResponse.userId)" -ForegroundColor Gray
    Write-Host "  Name: $($registerResponse.name)" -ForegroundColor Gray
    Write-Host "  Email: $($registerResponse.email)" -ForegroundColor Gray
} catch {
    $errorResponse = $_.ErrorDetails.Message | ConvertFrom-Json -ErrorAction SilentlyContinue
    if ($errorResponse.error -like "*already exists*") {
        Write-Host "Warning: User already exists (this is OK)" -ForegroundColor Yellow
    } else {
        Write-Host "Error: Registration failed: $($errorResponse.error)" -ForegroundColor Red
    }
}

# Test 2: Login with the user
Write-Host "`n2. Testing Login..." -ForegroundColor Yellow
$loginBody = @{
    email = "test@example.com"
    password = "test1234"
} | ConvertTo-Json

try {
    $loginResponse = Invoke-RestMethod -Uri "http://localhost:8082/api/auth/login" -Method POST -ContentType "application/json" -Body $loginBody
    Write-Host "Success: Login successful!" -ForegroundColor Green
    Write-Host "  Token: $($loginResponse.token.Substring(0,50))..." -ForegroundColor Gray
    $token = $loginResponse.token
    
    # Test 3: Get user info with token
    Write-Host "`n3. Testing /auth/me endpoint..." -ForegroundColor Yellow
    $headers = @{
        Authorization = "Bearer $token"
    }
    $meResponse = Invoke-RestMethod -Uri "http://localhost:8082/api/auth/me" -Method GET -Headers $headers
    Write-Host "Success: User info retrieved!" -ForegroundColor Green
    Write-Host "  User ID: $($meResponse.userId)" -ForegroundColor Gray
    Write-Host "  Name: $($meResponse.name)" -ForegroundColor Gray
    Write-Host "  Email: $($meResponse.email)" -ForegroundColor Gray
    
    Write-Host "`n=== ALL API TESTS PASSED ===" -ForegroundColor Green
    Write-Host "`nThe backend is working correctly." -ForegroundColor Cyan
    Write-Host "If login still fails in browser, check:" -ForegroundColor Yellow
    Write-Host "  - Browser console (F12 > Console tab)" -ForegroundColor White
    Write-Host "  - Network tab for failed requests" -ForegroundColor White
    Write-Host "  - CORS errors in console" -ForegroundColor White
    
} catch {
    $errorResponse = $_.ErrorDetails.Message | ConvertFrom-Json -ErrorAction SilentlyContinue
    Write-Host "Error: Login failed: $($errorResponse.error)" -ForegroundColor Red
}
