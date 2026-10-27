# Start the SlumberWell API server
# Run from the 'ML project' directory:
#
#   cd "C:\Users\TANVI\Downloads\ML project"
#   uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
#
# Then open: http://127.0.0.1:8000
# API docs:  http://127.0.0.1:8000/docs

Write-Host "Starting SlumberWell API..." -ForegroundColor Cyan
Set-Location "C:\Users\TANVI\Downloads\ML project"
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
