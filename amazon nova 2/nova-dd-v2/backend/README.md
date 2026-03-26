# Nova DueDiligence Backend (Local-first, Cloud-ready)

This backend is designed to run free on a single computer while staying ready to scale.

## Run locally

1. Create a `.env` file from `.env.example`.
2. Install dependencies:
   - `py -3.11 -m pip install -r requirements.txt`
3. Start API:
   - `./start.ps1`

## Storage modes

- `OBJECT_STORE_MODE=local`: stores PDFs and extracted text under `data/object_store`
- `OBJECT_STORE_MODE=s3`: stores files in `OBJECT_STORE_BUCKET`

Switching modes does not require API code changes.

## Free-tier guardrails

- `MAX_DOCS_TOTAL`
- `MAX_ANALYSES_PER_HOUR`
- `MAX_QUESTIONS_PER_HOUR`
- `MAX_UPLOAD_MB`

These limits prevent runaway Nova costs during development.
