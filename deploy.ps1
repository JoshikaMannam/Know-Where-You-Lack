# Deployment Script for Know Where You Lack
# This script helps prepare and deploy the application

param(
    [Parameter(Mandatory=$false)]
    [ValidateSet('check', 'prepare', 'deploy')]
    [string]$Action = 'check'
)

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Know Where You Lack - Deployment" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

function Check-Prerequisites {
    Write-Host "Checking prerequisites..." -ForegroundColor Yellow
    
    # Check Git
    try {
        $gitVersion = git --version
        Write-Host "✓ Git installed: $gitVersion" -ForegroundColor Green
    } catch {
        Write-Host "✗ Git not found. Please install Git." -ForegroundColor Red
        return $false
    }
    
    # Check Node
    try {
        $nodeVersion = node --version
        Write-Host "✓ Node.js installed: $nodeVersion" -ForegroundColor Green
    } catch {
        Write-Host "✗ Node.js not found. Please install Node.js 18+." -ForegroundColor Red
        return $false
    }
    
    # Check Java
    try {
        $javaVersion = java -version 2>&1 | Select-Object -First 1
        Write-Host "✓ Java installed: $javaVersion" -ForegroundColor Green
    } catch {
        Write-Host "✗ Java not found. Please install Java 21." -ForegroundColor Red
        return $false
    }
    
    # Check Maven
    try {
        $mvnVersion = mvn --version | Select-Object -First 1
        Write-Host "✓ Maven installed: $mvnVersion" -ForegroundColor Green
    } catch {
        Write-Host "✗ Maven not found. Please install Maven." -ForegroundColor Red
        return $false
    }
    
    return $true
}

function Check-EnvironmentFiles {
    Write-Host "`nChecking environment files..." -ForegroundColor Yellow
    
    $allGood = $true
    
    # Check .env (should exist but not be committed)
    if (Test-Path ".env") {
        Write-Host "✓ .env exists (contains secrets - won't be committed)" -ForegroundColor Green
    } else {
        Write-Host "✗ .env not found. Create it from .env.example" -ForegroundColor Red
        $allGood = $false
    }
    
    # Check .env.example
    if (Test-Path ".env.example") {
        Write-Host "✓ .env.example exists (will be committed)" -ForegroundColor Green
    } else {
        Write-Host "✗ .env.example not found" -ForegroundColor Red
        $allGood = $false
    }
    
    # Check frontend env files
    if (Test-Path "frontend\.env.development") {
        Write-Host "✓ frontend/.env.development exists" -ForegroundColor Green
    } else {
        Write-Host "✗ frontend/.env.development not found" -ForegroundColor Red
        $allGood = $false
    }
    
    if (Test-Path "frontend\.env.production") {
        Write-Host "✓ frontend/.env.production exists" -ForegroundColor Green
    } else {
        Write-Host "✗ frontend/.env.production not found" -ForegroundColor Red
        $allGood = $false
    }
    
    return $allGood
}

function Check-GitStatus {
    Write-Host "`nChecking Git status..." -ForegroundColor Yellow
    
    $status = git status --porcelain
    if ($status) {
        Write-Host "⚠ Uncommitted changes detected:" -ForegroundColor Yellow
        git status --short
        return $false
    } else {
        Write-Host "✓ Working directory clean" -ForegroundColor Green
        return $true
    }
}

function Prepare-Deployment {
    Write-Host "`nPreparing for deployment..." -ForegroundColor Yellow
    
    # Verify .env has required variables
    Write-Host "`nVerifying .env variables..." -ForegroundColor Yellow
    $envContent = Get-Content ".env" -Raw
    
    $requiredVars = @('DB_PASSWORD', 'GROQ_API_KEY', 'JWT_SECRET')
    $allPresent = $true
    
    foreach ($var in $requiredVars) {
        if ($envContent -match "$var=\S+") {
            Write-Host "✓ $var is set" -ForegroundColor Green
        } else {
            Write-Host "✗ $var is missing or empty" -ForegroundColor Red
            $allPresent = $false
        }
    }
    
    if (-not $allPresent) {
        Write-Host "`n✗ Please fill in all required environment variables in .env" -ForegroundColor Red
        return $false
    }
    
    # Test backend build
    Write-Host "`nTesting backend build..." -ForegroundColor Yellow
    Push-Location java-backend
    try {
        $buildResult = mvn clean package -DskipTests 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✓ Backend build successful" -ForegroundColor Green
        } else {
            Write-Host "✗ Backend build failed" -ForegroundColor Red
            Write-Host $buildResult
            Pop-Location
            return $false
        }
    } catch {
        Write-Host "✗ Backend build error: $_" -ForegroundColor Red
        Pop-Location
        return $false
    }
    Pop-Location
    
    # Test frontend build
    Write-Host "`nTesting frontend build..." -ForegroundColor Yellow
    Push-Location frontend
    try {
        if (-not (Test-Path "node_modules")) {
            Write-Host "Installing frontend dependencies..." -ForegroundColor Yellow
            npm install
        }
        
        $buildResult = npm run build 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✓ Frontend build successful" -ForegroundColor Green
        } else {
            Write-Host "✗ Frontend build failed" -ForegroundColor Red
            Write-Host $buildResult
            Pop-Location
            return $false
        }
    } catch {
        Write-Host "✗ Frontend build error: $_" -ForegroundColor Red
        Pop-Location
        return $false
    }
    Pop-Location
    
    return $true
}

function Deploy-Changes {
    Write-Host "`nCommitting changes for deployment..." -ForegroundColor Yellow
    
    # Show what will be committed
    Write-Host "`nFiles to be committed:" -ForegroundColor Cyan
    git status --short
    
    Write-Host "`nCommit message: " -ForegroundColor Cyan -NoNewline
    $commitMsg = Read-Host
    
    if ([string]::IsNullOrWhiteSpace($commitMsg)) {
        $commitMsg = "Prepare for production deployment"
    }
    
    # Add files
    git add .
    
    # Verify .env is not being committed
    $stagedFiles = git diff --cached --name-only
    if ($stagedFiles -match "^\.env$") {
        Write-Host "`n✗ ERROR: .env file is staged for commit!" -ForegroundColor Red
        Write-Host "This would expose your secrets. Unstaging..." -ForegroundColor Red
        git reset HEAD .env
        return $false
    }
    
    Write-Host "✓ .env is not being committed (secrets are safe)" -ForegroundColor Green
    
    # Commit
    git commit -m $commitMsg
    
    # Ask to push
    Write-Host "`nPush to GitHub? (y/n): " -ForegroundColor Yellow -NoNewline
    $push = Read-Host
    
    if ($push -eq 'y' -or $push -eq 'Y') {
        $branch = git branch --show-current
        Write-Host "Pushing to origin/$branch..." -ForegroundColor Yellow
        git push origin $branch
        Write-Host "✓ Pushed to GitHub" -ForegroundColor Green
        return $true
    } else {
        Write-Host "⚠ Changes committed locally but not pushed" -ForegroundColor Yellow
        return $false
    }
}

# Main execution
switch ($Action) {
    'check' {
        Write-Host "Running pre-deployment checks...`n" -ForegroundColor Cyan
        
        $prereqOk = Check-Prerequisites
        $envOk = Check-EnvironmentFiles
        $gitOk = Check-GitStatus
        
        Write-Host "`n========================================" -ForegroundColor Cyan
        if ($prereqOk -and $envOk) {
            Write-Host "✓ All checks passed!" -ForegroundColor Green
            Write-Host "`nNext steps:" -ForegroundColor Cyan
            Write-Host "1. Run: .\deploy.ps1 -Action prepare" -ForegroundColor White
            Write-Host "2. Then follow DEPLOYMENT_COMPLETE_GUIDE.md" -ForegroundColor White
        } else {
            Write-Host "✗ Some checks failed. Fix issues above." -ForegroundColor Red
        }
        Write-Host "========================================`n" -ForegroundColor Cyan
    }
    
    'prepare' {
        Write-Host "Preparing for deployment...`n" -ForegroundColor Cyan
        
        if (Check-Prerequisites -and Check-EnvironmentFiles) {
            if (Prepare-Deployment) {
                Write-Host "`n========================================" -ForegroundColor Cyan
                Write-Host "✓ Ready for deployment!" -ForegroundColor Green
                Write-Host "`nNext steps:" -ForegroundColor Cyan
                Write-Host "1. Run: .\deploy.ps1 -Action deploy" -ForegroundColor White
                Write-Host "2. Follow DEPLOYMENT_COMPLETE_GUIDE.md for Render setup" -ForegroundColor White
                Write-Host "========================================`n" -ForegroundColor Cyan
            } else {
                Write-Host "`n✗ Preparation failed. Fix errors above." -ForegroundColor Red
            }
        }
    }
    
    'deploy' {
        Write-Host "Deploying to GitHub...`n" -ForegroundColor Cyan
        
        if (Deploy-Changes) {
            Write-Host "`n========================================" -ForegroundColor Cyan
            Write-Host "✓ Code pushed to GitHub!" -ForegroundColor Green
            Write-Host "`nNext: Set up Render services" -ForegroundColor Cyan
            Write-Host "Follow: DEPLOYMENT_COMPLETE_GUIDE.md" -ForegroundColor White
            Write-Host "========================================`n" -ForegroundColor Cyan
        }
    }
}
