# Brain Tumor Classification - GitHub Push Automation Script

# Use system git directly if available, otherwise fallback to default path
$gitExe = "git"
try {
    $null = Get-Command $gitExe -ErrorAction Stop
}
catch {
    $gitExe = "C:\Program Files\Git\cmd\git.exe"
}

Write-Host "Using Git at: $gitExe" -ForegroundColor Cyan
if (!(Test-Path $gitExe) -and ($gitExe -ne "git")) {
    Write-Host "Error: Git not found." -ForegroundColor Red
    exit
}

function git_run {
    & $gitExe @args
}

Write-Host "`n[1/3] Preparing files..." -ForegroundColor Cyan
if (!(Test-Path .git)) {
    git_run init
}
git_run add .
git_run commit -m "Update MRI classification pipeline: Added safety validation and environment fixes"

Write-Host "`n[2/3] Configuring Remote..." -ForegroundColor Cyan
git_run branch -M main

$remotes = git_run remote
$repoUrl = "https://github.com/kdineshbabu988-code/MRI-Brain-Tumor-Classification.git"

if ($remotes -contains "origin") {
    git_run remote set-url origin $repoUrl
}
else {
    git_run remote add origin $repoUrl
}

Write-Host "`n[3/3] Pushing to GitHub..." -ForegroundColor Cyan
git_run push -u origin main

Write-Host "`nDone!" -ForegroundColor Green
