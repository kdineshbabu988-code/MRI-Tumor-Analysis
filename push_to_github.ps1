# Brain Tumor Classification - GitHub Push Automation Script

Write-Host "Checking for Git installation..." -ForegroundColor Cyan
if (!(Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "Error: Git is not installed or not in your PATH." -ForegroundColor Red
    Write-Host "Please install Git from https://git-scm.com/download/win and then run this script again." -ForegroundColor Yellow
    exit
}

Write-Host "Git found! Proceeding with GitHub push..." -ForegroundColor Green

# 1. Initialize and add files
Write-Host "`n[1/3] Initializing Git and adding files..." -ForegroundColor Cyan
git init
git add .
git commit -m "Initial commit: Production-ready MRI classification pipeline"

# 2. Configure Remote
Write-Host "`n[2/3] Linking to repository..." -ForegroundColor Cyan
git branch -M main
if (git remote | findstr "origin") {
    git remote remove origin
}
git remote add origin https://github.com/kdineshbabu988-code/MRI-Brain-Tumor-Classification.git

# 3. Push
Write-Host "`n[3/3] Pushing to GitHub (this may ask for your login)..." -ForegroundColor Cyan
git push -u origin main

Write-Host "`nDone! If you saw a login prompt, the push was successful." -ForegroundColor Green
