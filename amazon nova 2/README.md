# Nova DueDiligence

**AI-powered document intelligence for business professionals.**  
Built on Amazon Nova Pro via AWS Bedrock — Amazon Nova Hackathon submission.

---

## Overview

Due diligence is one of the most time-intensive workflows in business and legal practice. A mid-market M&A transaction typically involves reviewing 50–200 documents — contracts, NDAs, vendor agreements, financial statements — before a deal can close. Analysts and legal teams spend days on first-pass review, at billing rates of $400–$800 per hour.

Nova DueDiligence compresses that process to minutes.

Upload any business PDF. A parallel pipeline of four specialist AI agents — each powered by Amazon Nova Pro — simultaneously analyzes the document for risks, financial exposure, party obligations, and compliance gaps. Results are returned with structured findings, confidence scores on every item, and a 0–100 risk score. A board-ready executive memo can be generated and downloaded as a formatted PDF with one click.

---

## What It Does

### Deep Analysis — Parallel Multi-Agent Pipeline

The core feature. One document in, full structured analysis out.

An Orchestrator Agent classifies the document and dispatches four specialist agents simultaneously via Python `ThreadPoolExecutor`. Each agent runs as an independent Amazon Nova Pro call with a domain-expert system prompt:

| Agent | Specialization |
|---|---|
| **Risk Agent** | Liability exposure, unfavorable clauses, legal risk flags |
| **Financial Agent** | Payment terms, penalties, total financial exposure |
| **Obligations Agent** | Party obligations, deadlines, performance requirements |
| **Compliance Agent** | Missing standard protections, regulatory gaps |

All four agents run in parallel. Wall-clock time is bounded by the slowest agent — not the sum of all four. The Orchestrator synthesizes all findings into a unified risk score (0–100) with an overall confidence rating.

Every finding includes a confidence score. Analysts can triage by severity and confidence before acting.

### Document Q&A — Semantic Search

Ask any question in plain language across one or more documents simultaneously.

On upload, every document is chunked and embedded using Amazon Titan Embed Text v2 (512 dimensions). Questions are embedded at query time and cosine similarity search identifies the most relevant passages before passing them to Nova Pro for answer generation. Answers include source citations and retrieval metadata.

For large documents that exceed embedding rate limits, the system automatically switches to full-text retrieval using Nova Pro's 300,000 token context window — no chunking required for typical legal documents.

### Cross-Document Comparison

Select two or more documents to identify conflicts, inconsistencies, and misaligned terms across your document set. Nova Pro analyzes the full text of all selected documents in a single call, enabling genuine cross-document reasoning. Conflicts are returned with severity ratings, the specific documents involved, and recommended resolutions.

### Risk Dashboard

Aggregate risk metrics across all analyzed documents. Includes a 0–100 risk gauge, category breakdown by Legal / Financial / Operational / Compliance, severity distribution, and per-document scores. Designed for portfolio-level review before a deal meeting.

### Executive Report

One click generates a structured due diligence memorandum covering all risk findings, obligations, financial exposure, compliance gaps, and prioritized recommendations. Downloadable as a formatted PDF ready for stakeholder distribution.

---

## Architecture

```
┌─────────────────────────────────────────────┐
│         Frontend (HTML + React + Chart.js)  │
│Dashboard · Analysis · Q&A · Compare · Report│
└─────────────────────┬───────────────────────┘
                      │ REST API
┌─────────────────────▼───────────────────────┐
│        FastAPI Backend (Python 3.11)        │
│                                             │
│  ┌───────────────────────────────────────┐  │
│  │         Orchestrator Agent            │  │
│  │   Classifies document, routes work    │  │
│  └──┬─────────┬──────────┬───────────┬───┘  │
│     │         │          │           │      │
│  ┌──▼───┐ ┌───▼───┐ ┌────▼────┐ ┌────▼──┐   │
│  │ Risk │ │ Fin.  │ │ Oblig.  │ │Comply │   │
│  │Agent │ │ Agent │ │  Agent  │ │ Agent │   │
│  └──────┘ └───────┘ └─────────┘ └───────┘   │
│         [All 4 run in parallel]             │
│                                             │
│  ┌────────────────────────────────────────┐ │
│  │             Vector Store               │ │
│  │  Titan Embed v2 · Cosine similarity    │ │
│  │  Auto-switch: parallel / sequential    │ │
│  └────────────────────────────────────────┘ │
└─────────────────────┬───────────────────────┘
                      │
┌─────────────────────▼───────────────────────┐
│                 AWS Bedrock                 │
│   amazon.nova-pro-v1:0  (us cross-region)   │
│   amazon.titan-embed-text-v2:0              │
└─────────────────────────────────────────────┘
```

**Frontend:** Single HTML file — React via CDN, Chart.js for visualizations, jsPDF for report export. No build step. Opens directly in any browser.

**Backend:** Python 3.11, FastAPI, pypdf for text extraction, python-dotenv for credential management.

**AWS Services:** Amazon Bedrock (model inference), Amazon Nova Pro (all reasoning and generation), Amazon Titan Embed Text v2 (document embeddings).

---

## Key Technical Details

**Parallel agent execution.** The four specialist agents run concurrently. At typical Nova Pro latencies of 8–15 seconds per call, parallel execution saves approximately 24–45 seconds per document compared to sequential execution. Measurable by timing the `/analyze` endpoint directly.

**300,000 token context window.** A 50-page legal contract runs approximately 25,000–40,000 tokens. Nova Pro ingests the full document in a single call — no chunking required at analysis time. Cross-references between sections are preserved.

**Confidence scores on every finding.** Each risk flag, financial term, obligation, and missing clause includes a `confidence` field (0.0–1.0). Surfaced as visual indicators in the UI, enabling analysts to triage uncertain findings from high-confidence ones.

**Auto-switching embedding.** Documents under 50 chunks use parallel embedding (fast). Larger documents automatically switch to rate-limited sequential embedding with a user-visible notification. The system never fails silently — fallback mode is always disclosed.

**Agent reasoning trace.** Every analysis returns a full step-by-step trace log showing each agent's actions, item counts, and confidence. Displayed in the UI as an animated panel. All AI decisions are auditable.

---

## Setup

### Requirements

- Python 3.11
- AWS account with Bedrock access
- Amazon Nova Pro (`us.amazon.nova-pro-v1:0`) — enabled automatically on first invocation
- Amazon Titan Embed Text v2 (`amazon.titan-embed-text-v2:0`) — invoke once to activate

### Installation

```bash
cd backend
py -3.11 -m pip install -r requirements.txt
```

### Configuration

Create `backend/.env`:

```
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
AWS_REGION=your_region_here
```

Or create `backend/start.ps1` (Windows — recommended):

```powershell
$env:AWS_ACCESS_KEY_ID="your_access_key_here"
$env:AWS_SECRET_ACCESS_KEY="your_secret_key_here"
$env:AWS_REGION="your_region_here"
py -3.11 -m uvicorn main:app --port 8000
```

### Running

```bash
py -3.11 -m uvicorn main:app --port 8000
```

Open `frontend/index.html` in Chrome. Click **Configure** (top right) and confirm the URL is `http://localhost:8000`.

Verify connectivity:

```
http://localhost:8000/health
```

Expected response:

```json
{
  "status": "ok",
  "model": "amazon.nova-pro-v1:0",
  "embed_model": "amazon.titan-embed-text-v2:0",
  "agent_mode": "parallel",
  "context_window": "300K tokens"
}
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | System status and model info |
| GET | `/documents` | List all uploaded documents |
| POST | `/upload` | Upload PDF — extract, summarize, embed |
| POST | `/analyze/{id}` | Run full 4-agent parallel pipeline |
| POST | `/ask` | Semantic RAG Q&A across selected documents |
| POST | `/compare` | Cross-document conflict analysis |
| POST | `/report` | Generate executive due diligence memo |
| POST | `/dashboard` | Aggregate risk metrics across documents |
| DELETE | `/document/{id}` | Remove document and embeddings |

---

## File Structure

```
nova-duediligence/
├── frontend/
│   └── index.html          Single-file React application
├── backend/
│   ├── main.py             FastAPI server — all endpoints
│   ├── agents.py           Orchestrator + 4 specialist agents
│   ├── embeddings.py       Titan Embed v2 + vector store + auto-switch logic
│   ├── requirements.txt    Python dependencies
│   └── .env                AWS credentials (create manually)
└── README.md
```

---

## Known Limitations

- **Scanned PDFs not supported.** Text extraction requires selectable text. Image-based PDFs return no content.
- **In-memory storage.** Documents are held in process memory. Restarting the server clears all data. A production deployment would use a database.
- **No authentication.** All documents are accessible to any client that can reach the server. Production use requires an authentication layer.
- **Human review required.** Nova Pro, like all large language models, can produce plausible-sounding but incorrect outputs. All findings should be reviewed by a qualified analyst or legal professional before being acted upon.
- **New AWS accounts subject to rate limits.** Titan Embed calls are rate-limited on new accounts. The auto-switch logic handles this transparently.

---

## Hackathon Details

| Field | Value |
|---|---|
| Event | Amazon Nova Hackathon |
| Category | Agentic AI |
| Nova Models | `us.amazon.nova-pro-v1:0`, `amazon.titan-embed-text-v2:0` |
| AWS Services | Amazon Bedrock |
| Hashtag | #AmazonNova |