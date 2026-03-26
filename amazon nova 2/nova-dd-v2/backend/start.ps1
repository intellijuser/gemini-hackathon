if (-Not (Test-Path ".env")) {
    Write-Host ".env not found. Copy .env.example to .env and fill values."
}

py -3.11 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
