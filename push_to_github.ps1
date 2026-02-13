# Brain Tumor Classification - GitHub Push Automation Script

$gitPath = "C:\Program Files\Git\cmd\git.exe"
Write-Host "Checking for Git installation at $gitPath..." -ForegroundColor Cyan
if (!(Test-Path $gitPath)) {
    Write-Host "Error: Git not found at $gitPath." -ForegroundColor Red
    Write-Host "Please install Git or update the path in this script." -ForegroundColor Yellow
    exit
}

function git_run {
    & $gitPath @args
}

Write-Host "Git found! Proceeding with GitHub push..." -ForegroundColor Green

# 1. Initialize and add files
Write-Host "`n[1/3] Initializing Git and adding files..." -ForegroundColor Cyan
git_run init
git_run add .
git_run commit -m "Initial commit: Production-ready MRI classification pipeline"

# 2. Configure Remote
Write-Host "`n[2/3] Linking to repository..." -ForegroundColor Cyan
git_run branch -M main
if (git_run remote | findstr "origin") {
    git_run remote remove origin
}
git_run remote add origin https://github.com/kdineshbabu988-code/MRI-Brain-Tumor-Classification.git

# 3. Push
Write-Host "`n[3/3] Pushing to GitHub (this may ask for your login)..." -ForegroundColor Cyan
git_run push -u origin main

Write-Host "`nDone! If you saw a login prompt, the push was successful." -ForegroundColor Green
