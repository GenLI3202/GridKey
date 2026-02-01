# GridKey Optimizer Service - Startup Script for Windows
# =====================================================
#
# Usage:
#   .\startup.ps1          - Run with default configuration
#   .\startup.ps1 dev      - Run with hot-reload
#   .\startup.ps1 test     - Run tests

$ErrorActionPreference = "Stop"

# Default configuration
$Host = "0.0.0.0"
$Port = "8000"
$Reload = $false

# Parse arguments
switch ($args[0]) {
    "dev" {
        Write-Host "Mode: Development (hot-reload enabled)" -ForegroundColor Yellow
        $Reload = $true
    }
    "test" {
        Write-Host "Mode: Test Suite" -ForegroundColor Yellow
        Write-Host ""
        pytest src/test/ -v --tb=short
        exit $LASTEXITCODE
    }
    default {
        Write-Host "Mode: Production" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "======================================" -ForegroundColor Green
Write-Host "  GridKey Optimizer Service" -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Green
Write-Host ""
Write-Host "Host: $Host"
Write-Host "Port: $Port"
Write-Host "Reload: $Reload"
Write-Host ""

# Check if uvicorn is installed
try {
    $null = Get-Command uvicorn -ErrorAction Stop
} catch {
    Write-Host "Error: uvicorn not found" -ForegroundColor Red
    Write-Host "Install with: pip install uvicorn[standard]"
    exit 1
}

# Build uvicorn command
$uvicornCmd = "uvicorn src.api.main:app --host $Host --port $Port"
if ($Reload) {
    $uvicornCmd += " --reload"
}

# Start server
Write-Host "Starting server..." -ForegroundColor Green
Write-Host "API will be available at: http://$Host`:$Port" -ForegroundColor Green
Write-Host "Docs at: http://$Host`:$Port/docs" -ForegroundColor Green
Write-Host ""

Invoke-Expression $uvicornCmd
