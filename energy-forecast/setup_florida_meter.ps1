# Quick Setup Script for Florida Meter Data
# Run this to get started!

Write-Host "="*80 -ForegroundColor Cyan
Write-Host "FLORIDA ENERGY METER SETUP" -ForegroundColor Cyan
Write-Host "="*80 -ForegroundColor Cyan

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

Write-Host "`n📊 Step 1: Enter Your Meter Readings" -ForegroundColor Yellow
Write-Host "Let's add some real data from your Florida home/business`n"

# Run manual entry
.\.venv\Scripts\python.exe scripts\fetch_utility_data.py

Write-Host "`n`n✅ Step 2: Your backend will now use real data!" -ForegroundColor Green
Write-Host "The backend automatically detects and uses your meter readings.`n"

Write-Host "`n📈 Step 3: Compare Model Predictions vs Reality" -ForegroundColor Yellow
Write-Host "Press any key to see how accurate the models are...`n"
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

.\.venv\Scripts\python.exe scripts\compare_predictions.py

Write-Host "`n`n🎯 Next Steps:" -ForegroundColor Cyan
Write-Host "  1. Open Streamlit: http://localhost:8501"
Write-Host "  2. Click 'Get Forecast' - now using YOUR data!"
Write-Host "  3. Download more data from your utility for better accuracy"
Write-Host "`n  Florida Utilities:"
Write-Host "    - FPL: https://www.fpl.com/energy-dashboard"
Write-Host "    - Duke: https://www.duke-energy.com/myaccount"
Write-Host "    - TECO: https://www.tampaelectric.com/myaccount"
Write-Host "`n📖 Full guide: docs\FLORIDA_METER_GUIDE.md`n"

Write-Host "="*80 -ForegroundColor Cyan
Write-Host "Setup Complete! Your models are now using real Florida meter data." -ForegroundColor Green
Write-Host "="*80 -ForegroundColor Cyan
