"""
Nova DueDiligence — FastAPI Backend (Production)
All state persisted in Postgres + pgvector. Analysis via Celery.
Rate-limited. Workspace-scoped. No in-memory dicts.
"""
import os
import json
import logging
import time as _time
import google.generativeai as genai
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text as sa_text
from prometheus_client import make_asgi_app, Counter, Histogram
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from starlette.responses import JSONResponse

from config import settings
from database import engine, Base, get_async_session
from auth import fastapi_users_app, auth_backend, current_active_user
from schemas_auth import UserRead, UserCreate
from workspaces import router as workspaces_router
from agents import OrchestratorAgent, call_gemini
from embeddings import embed_text, embed_query, chunk_text_with_metadata
from tasks import process_document_task
from celery.result import AsyncResult
import models  # register models
import crud

logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","name":"%(name)s","level":"%(levelname)s","msg":"%(message)s"}',
)
logger = logging.getLogger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.execute(sa_text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables ensured.")
    yield

app = FastAPI(title="Nova DueDiligence API — Gemini Edition", version="5.1.0", lifespan=lifespan)

# ── Rate Limiter ──────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded. Try again later."})

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Auth + Workspace Routes ──────────────────────────────────────────────────
app.include_router(fastapi_users_app.get_auth_router(auth_backend), prefix="/auth/jwt", tags=["auth"])
app.include_router(fastapi_users_app.get_register_router(UserRead, UserCreate), prefix="/auth", tags=["auth"])
app.include_router(workspaces_router)

# ── Gemini Client ────────────────────────────────────────────────────────────
genai.configure(api_key=settings.GEMINI_API_KEY)
orchestrator = OrchestratorAgent(model_name=settings.GEMINI_MODEL)

# ── Prometheus ────────────────────────────────────────────────────────────────
REQUEST_COUNT = Counter("http_requests_total", "Total HTTP requests", ["method", "endpoint", "http_status"])
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds", "Request latency",
    ["endpoint"], buckets=[0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30],
)
ANALYSIS_COST = Counter("gemini_analysis_cost_usd", "Estimated analysis cost in USD")

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    if request.url.path.startswith("/metrics"):
        return await call_next(request)
    t0 = _time.time()
    try:
        response = await call_next(request)
        latency = _time.time() - t0
        REQUEST_LATENCY.labels(endpoint=request.url.path).observe(latency)
        REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path, http_status=response.status_code).inc()
        response.headers["X-Response-Time"] = f"{round(latency * 1000)}ms"
        return response
    except Exception as e:
        REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path, http_status=500).inc()
        raise


# ── Request Models ────────────────────────────────────────────────────────────
class QuestionRequest(BaseModel):
    question: str
    document_ids: list[str]
    workspace_id: str

class DocListRequest(BaseModel):
    document_ids: list[str]
    workspace_id: str


# ── Helpers ───────────────────────────────────────────────────────────────────
_server_start = _time.time()

def risk_label(score: int) -> str:
    if score >= 75: return "Critical"
    if score >= 50: return "High"
    if score >= 28: return "Medium"
    return "Low"


# ── Health (real DB stats) ────────────────────────────────────────────────────
@app.get("/health")
async def health(session: AsyncSession = Depends(get_async_session)):
    try:
        await session.execute(sa_text("SELECT 1"))
        pg_status = "connected"
    except Exception:
        pg_status = "disconnected"

    doc_counts = await crud.count_docs(session)
    chunk_count = await crud.count_chunks(session)

    # P95 from Prometheus histogram
    secs = int(_time.time() - _server_start)
    h, r = divmod(secs, 3600)
    m, s = divmod(r, 60)

    return {
        "status": "ok",
        "version": "5.0.0",
        "postgres": pg_status,
        "model": settings.GEMINI_MODEL,
        "embed_model": settings.GEMINI_EMBED_MODEL,
        "uptime": f"{h}h {m}m {s}s",
        "documents": doc_counts,
        "chunks_indexed": chunk_count,
    }


# ── Upload (workspace-scoped, persisted to DB) ───────────────────────────────
@app.post("/upload/{workspace_id}")
@limiter.limit(f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
async def upload_document(
    request: Request,
    workspace_id: str,
    file: UploadFile = File(...),
    user=Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
):
    if not await crud.verify_workspace_access(session, workspace_id, user.id):
        raise HTTPException(403, "Access denied to workspace")

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")

    file_bytes = await file.read()
    if len(file_bytes) > settings.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(400, f"File too large (max {settings.MAX_UPLOAD_MB}MB).")

    # Create doc record in Postgres
    doc = await crud.create_doc(session, workspace_id, file.filename)

    # Persist file to disk
    os.makedirs("uploads", exist_ok=True)
    file_path = f"uploads/{doc.id}.pdf"
    with open(file_path, "wb") as f:
        f.write(file_bytes)

    logger.info(f"Uploaded doc={doc.id} file={file.filename} workspace={workspace_id}")
    return {"id": doc.id, "filename": doc.filename, "status": doc.status}


# ── Analyze (Celery async, idempotent) ────────────────────────────────────────
@app.post("/analyze/{document_id}")
@limiter.limit(f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
async def analyze_document(
    request: Request,
    document_id: str,
    user=Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
):
    doc = await crud.get_doc(session, document_id)
    if not doc:
        raise HTTPException(404, "Document not found.")
    if not await crud.verify_workspace_access(session, doc.workspace_id, user.id):
        raise HTTPException(403, "Access denied")

    # Idempotent: skip if already analyzed
    if doc.status == "analyzed":
        analysis = await crud.get_analysis(session, document_id)
        return {
            "document_id": document_id,
            "status": "already_analyzed",
            "risk_score": analysis.risk_score if analysis else None,
        }

    # Idempotent: skip if already processing
    if doc.status == "processing":
        return {"document_id": document_id, "status": "processing"}

    task = process_document_task.delay(doc.id)
    return {"job_id": task.id, "document_id": document_id, "status": "queued"}


# ── Job Status ────────────────────────────────────────────────────────────────
@app.get("/status/{job_id}")
async def get_task_status(job_id: str):
    task_result = AsyncResult(job_id)
    return {
        "job_id": job_id,
        "status": task_result.status,
        "result": task_result.result if task_result.ready() else None,
    }


# ── Ask (pgvector semantic search) ───────────────────────────────────────────
@app.post("/ask")
@limiter.limit(f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
async def ask_question(
    request: Request,
    req: QuestionRequest,
    user=Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
):
    if not req.document_ids:
        raise HTTPException(400, "No documents selected.")
    if not await crud.verify_workspace_access(session, req.workspace_id, user.id):
        raise HTTPException(403, "Access denied")

    docs = await crud.get_docs_by_ids(session, req.document_ids)
    doc_map = {d.id: d for d in docs}
    missing = [did for did in req.document_ids if did not in doc_map]
    if missing:
        raise HTTPException(404, f"Documents not found: {missing}")

    # pgvector search
    query_emb = embed_query(req.question)
    if query_emb:
        chunks = await crud.search_chunks_pgvector(
            session, query_emb, doc_ids=req.document_ids,
            top_k=settings.SEMANTIC_TOP_K, threshold=settings.SEMANTIC_THRESHOLD,
        )
        using_semantic = bool(chunks)
    else:
        chunks = []
        using_semantic = False

    doc_names = [doc_map[did].filename for did in req.document_ids if did in doc_map]

    if using_semantic:
        context_parts = []
        for c in chunks:
            source = doc_map.get(c["doc_id"])
            source_name = source.filename if source else c["doc_id"]
            page = f" | Page ~{c.get('page_estimate', '?')}" if c.get("page_estimate") else ""
            section = f" | {c.get('section_heading', '')}" if c.get("section_heading") else ""
            header = f"[Source: {source_name}{page}{section} | Relevance: {c['score']:.0%}]"
            context_parts.append(f"{header}\n{c['chunk']}")
        context = "\n\n---\n\n".join(context_parts)
        method = "pgvector_semantic"
    else:
        # Fallback: no embeddings available
        context = "\n\n".join(
            f'=== {doc_map[did].filename} ===\n{doc_map[did].summary or ""}'
            for did in req.document_ids if did in doc_map
        )
        method = "full_text_fallback"

    answer = call_gemini(
        """You are a senior legal and financial analyst with expertise in M&A due diligence.
Answer using ONLY the provided document context — never invent facts.
Always cite the document name and section number for every claim.
If the answer is not in the context, say "This information is not available in the selected documents".""",
        f"Documents: {', '.join(doc_names)}\n\nCONTEXT:\n{context}\n\nQUESTION: {req.question}\n\nAnswer with specific citations:",
        1500,
    )

    return {
        "question": req.question,
        "answer": answer,
        "documents_used": doc_names,
        "retrieval_method": method,
        "chunks_retrieved": len(chunks),
        "top_relevance_score": round(chunks[0]["score"], 3) if chunks else 0,
    }


# ── Compare ───────────────────────────────────────────────────────────────────
@app.post("/compare")
@limiter.limit(f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
async def compare_documents(
    request: Request,
    req: DocListRequest,
    user=Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
):
    if len(req.document_ids) < 2:
        raise HTTPException(400, "Select at least 2 documents.")
    if not await crud.verify_workspace_access(session, req.workspace_id, user.id):
        raise HTTPException(403, "Access denied")

    docs = await crud.get_docs_by_ids(session, req.document_ids)
    doc_map = {d.id: d for d in docs}
    doc_names = [doc_map[did].filename for did in req.document_ids if did in doc_map]

    # Get summaries for comparison — we don't store full text in DB anymore
    context = "\n\n".join(
        f'=== {doc_map[did].filename} ===\n{doc_map[did].summary or "No summary available"}'
        for did in req.document_ids if did in doc_map
    )

    raw = call_gemini(
        "You are a senior M&A due diligence specialist. Respond ONLY with valid JSON. No markdown fences.",
        f'''Compare these {len(doc_names)} documents and identify ALL conflicts, inconsistencies, and alignment issues.

Documents: {", ".join(doc_names)}

{context[:settings.CONTEXT_WINDOW_CHARS]}

Return this exact JSON:
{{
  "conflicts": [{{"title":"...","description":"...","documents_involved":["..."],"severity":"low|medium|high|critical","recommendation":"..."}}],
  "consistencies": ["..."],
  "cross_document_risks": [{{"risk":"...","severity":"low|medium|high"}}],
  "recommendations": ["..."],
  "overall_alignment": "poor|fair|good|excellent"
}}''',
        3000,
    )

    raw = raw.strip().replace("```json", "").replace("```", "").strip()
    try:
        parsed = json.loads(raw)
    except Exception:
        s, e = raw.find("{"), raw.rfind("}") + 1
        try:
            parsed = json.loads(raw[s:e]) if s >= 0 and e > s else {}
        except Exception:
            parsed = {"conflicts": [], "recommendations": [raw], "overall_alignment": "unknown"}

    return {"documents": doc_names, "comparison": parsed}


# ── Report ────────────────────────────────────────────────────────────────────
@app.post("/report")
@limiter.limit(f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
async def generate_report(
    request: Request,
    req: DocListRequest,
    user=Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
):
    if not req.document_ids:
        raise HTTPException(400, "No documents selected.")
    if not await crud.verify_workspace_access(session, req.workspace_id, user.id):
        raise HTTPException(403, "Access denied")

    docs = await crud.get_docs_by_ids(session, req.document_ids)
    doc_map = {d.id: d for d in docs}
    doc_names = [doc_map[did].filename for did in req.document_ids if did in doc_map]

    pre_analyses = ""
    for did in req.document_ids:
        analysis = await crud.get_analysis(session, did)
        if analysis:
            doc = doc_map.get(did)
            pre_analyses += (
                f'\n[PRE-ANALYZED: {doc.filename if doc else did} | '
                f'Risk: {analysis.risk_score}/100 ({analysis.risk_label}) | '
                f'Flags: {len(analysis.risk_flags or [])} | '
                f'{analysis.executive_summary or ""}]\n'
            )

    context = "\n\n".join(
        f'=== {doc_map[did].filename} ===\n{doc_map[did].summary or ""}'
        for did in req.document_ids if did in doc_map
    )

    report = call_gemini(
        """You are a Managing Director at a bulge-bracket investment bank preparing a due diligence memorandum.
Write formally, be specific, cite exact clause numbers, name exact parties, state exact dollar amounts.
Use numbered sections. This memo will be read by partners before a deal decision.""",
        f"""Prepare a comprehensive Executive Due Diligence Report for: {', '.join(doc_names)}

{pre_analyses}

CONTEXT:
{context[:settings.CONTEXT_WINDOW_CHARS]}

Required sections:
1. EXECUTIVE SUMMARY
2. SCOPE & DOCUMENTS REVIEWED
3. CRITICAL RISK FINDINGS
4. KEY OBLIGATIONS & DEADLINES
5. FINANCIAL EXPOSURE ANALYSIS
6. COMPLIANCE GAPS
7. RECOMMENDATIONS
8. OVERALL RISK RATING""",
        4096,
    )

    return {"documents": doc_names, "report": report}


# ── Dashboard ─────────────────────────────────────────────────────────────────
@app.post("/dashboard")
async def get_dashboard(
    req: DocListRequest,
    user=Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
):
    if not await crud.verify_workspace_access(session, req.workspace_id, user.id):
        raise HTTPException(403, "Access denied")

    per_doc = []
    all_flags = []
    total_score = 0
    analyzed_count = 0

    for did in req.document_ids:
        doc = await crud.get_doc(session, did)
        analysis = await crud.get_analysis(session, did)
        if doc and analysis:
            analyzed_count += 1
            flags = analysis.risk_flags or []
            all_flags.extend(flags)
            total_score += analysis.risk_score or 0
            per_doc.append({
                "filename": doc.filename,
                "risk_score": analysis.risk_score,
                "risk_label": analysis.risk_label,
                "flags": len(flags),
            })

    if not analyzed_count:
        return {"message": "No analyzed documents. Run Deep Analysis first.", "data": None}

    avg_score = int(total_score / analyzed_count)

    cat_totals = {"Legal": 0, "Financial": 0, "Operational": 0, "Compliance": 0}
    sev_totals = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for f in all_flags:
        cat = f.get("category", "Legal")
        if cat in cat_totals:
            cat_totals[cat] += 1
        sev = f.get("severity", "low").lower()
        if sev in sev_totals:
            sev_totals[sev] += 1

    top_risks = sorted(
        all_flags,
        key=lambda f: (
            {"critical": 4, "high": 3, "medium": 2, "low": 1}.get(f.get("severity", "low"), 1),
            float(f.get("confidence", 0)),
        ),
        reverse=True,
    )[:5]

    return {
        "aggregate_risk_score": avg_score,
        "risk_label": risk_label(avg_score),
        "documents_analyzed": analyzed_count,
        "total_risk_flags": len(all_flags),
        "category_breakdown": cat_totals,
        "severity_distribution": sev_totals,
        "top_risks": top_risks,
        "per_document": per_doc,
    }


# ── Delete Document ───────────────────────────────────────────────────────────
@app.delete("/document/{document_id}")
async def delete_document(
    document_id: str,
    user=Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
):
    doc = await crud.get_doc(session, document_id)
    if not doc:
        raise HTTPException(404, "Document not found.")
    if not await crud.verify_workspace_access(session, doc.workspace_id, user.id):
        raise HTTPException(403, "Access denied")

    await crud.delete_doc(session, document_id)

    # Clean up file
    file_path = f"uploads/{document_id}.pdf"
    if os.path.exists(file_path):
        os.remove(file_path)

    return {"deleted": document_id}