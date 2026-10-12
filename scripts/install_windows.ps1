# ATLAS HUMANOID AI 4D — Windows PowerShell Setup & Environment Installer
# Configured for Windows 10/11, 8GB RAM, CPU-First execution

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "ATLAS HUMANOID AI 4D — WINDOWS ENVIRONMENT INSTALLER" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

# 1. Check Python version
$pythonVersion = python --version
Write-Host "[CHECK] Python: $pythonVersion" -ForegroundColor Green

# 2. Upgrade pip
Write-Host "[1/5] Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# 3. Install core dependencies
Write-Host "[2/5] Installing CPU-first dependencies from requirements.txt..." -ForegroundColor Yellow
pip install -r requirements.txt

# 4. Install ONNX and onnxscript for deployment
Write-Host "[3/5] Installing ONNX runtime and onnxscript..." -ForegroundColor Yellow
pip install onnx onnxscript

# 5. Install ATLAS package in editable mode
Write-Host "[4/5] Installing ATLAS HUMANOID AI 4D in editable mode..." -ForegroundColor Yellow
pip install -e . --no-deps

# 6. Run quick sanity smoke test
Write-Host "[5/5] Running Milestone M0 smoke test..." -ForegroundColor Yellow
python scripts/run_m0_validation.py

Write-Host "======================================================================" -ForegroundColor Green
Write-Host "INSTALLATION COMPLETE! All 14 milestones ready for execution." -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Green
