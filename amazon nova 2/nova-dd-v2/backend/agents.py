"""
Nova DueDiligence — Parallel Multi-Agent Pipeline
Orchestrator + 4 Specialist Agents running concurrently
Powered by Google Gemini via Google AI SDK

Gemini migration:
- call_nova → call_gemini (google-generativeai SDK)
- Model: gemini-2.0-flash (fast, cheap) — swap to gemini-1.5-pro for max quality
- All agent prompts, scoring, synthesis, dedup unchanged
- Retry + exponential backoff preserved
"""
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import google.generativeai as genai
from google.generativeai.types import GenerationConfig

from schemas import (
    RiskAgentOutput, FinancialAgentOutput, ObligationsAgentOutput,
    ComplianceAgentOutput, validate_agent_output
)

logger = logging.getLogger(__name__)


def init_gemini(api_key: str):
    """Call once at startup to configure the Gemini SDK."""
    genai.configure(api_key=api_key)


def call_gemini(
    system: str,
    user: str,
    max_tokens: int = 2048,
    retries: int = 3,
    model_name: str = "gemini-2.0-flash",
) -> str:
    """
    Call Google Gemini with exponential backoff retry.
    system = system instruction (Gemini supports this natively)
    user   = user message content
    """
    last_error = None
    for attempt in range(retries):
        try:
            model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=system,
                generation_config=GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=0.1,
                    candidate_count=1,
                ),
            )
            response = model.generate_content(user)
            return response.text
        except Exception as e:
            last_error = e
            if attempt < retries - 1:
                wait = 2 ** attempt
                logger.warning(
                    f"Gemini call failed (attempt {attempt+1}/{retries}), "
                    f"retrying in {wait}s: {e}"
                )
                time.sleep(wait)
    raise last_error


def parse_json(text: str) -> dict:
    """Safely parse JSON from Gemini response. Handles markdown fences."""
    clean = text.strip()
    for fence in ["```json", "```JSON", "```"]:
        clean = clean.replace(fence, "")
    clean = clean.strip()
    try:
        return json.loads(clean)
    except Exception:
        pass
    s, e = clean.find("{"), clean.rfind("}") + 1
    if s >= 0 and e > s:
        try:
            return json.loads(clean[s:e])
        except Exception:
            pass
    return {}


def _semantic_dedup(flags: list) -> list:
    """Remove semantically duplicate flags by title word overlap."""
    if not flags:
        return flags
    seen = []
    deduped = []
    for f in flags:
        title_words = set(f.get("title", "").lower().split())
        is_dup = any(
            len(title_words & s_words) >= 2 and len(title_words) > 0
            for s_words in seen
        )
        if not is_dup:
            seen.append(title_words)
            deduped.append(f)
    return deduped


# ── Specialist Agents ─────────────────────────────────────────────────────────

class RiskSpecialistAgent:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.name = "Risk Agent"

    def analyze(self, text: str, filename: str) -> dict:
        raw = call_gemini(
            system="""You are a senior M&A attorney with 20 years of experience at a top law firm.
You have reviewed thousands of contracts for PE funds, corporations, and startups.
Think step by step: (1) identify parties and positions, (2) find asymmetric risk clauses,
(3) flag missing protections, (4) assess severity and likelihood.
Respond ONLY with valid JSON. No markdown fences, no explanation outside the JSON.""",
            user=f'''Analyze this document for legal and contractual risks.

DOCUMENT NAME: {filename}
DOCUMENT TEXT:
{text[:22000]}

Return ONLY this JSON:
{{
  "executive_summary": "3 precise sentences: document type and parties, most dangerous clause, overall risk posture and action",
  "parties": ["Full legal name of party 1", "Full legal name of party 2"],
  "risk_flags": [
    {{
      "title": "Concise unique risk title (5-8 words)",
      "description": "Precise description: what clause says, why it creates risk, financial consequence",
      "severity": "low|medium|high|critical",
      "category": "Legal|Financial|Operational|Compliance",
      "clause_reference": "Section X.Y or Not specified",
      "confidence": 0.95
    }}
  ],
  "confidence": 0.90
}}

Rules: Max 8 flags. Every flag UNIQUE. critical = >$1M exposure or termination risk.''',
            max_tokens=2800,
            model_name=self.model_name,
        )
        result = parse_json(raw)
        if not result:
            result = {"executive_summary": "Risk analysis failed.", "parties": [], "risk_flags": [], "confidence": 0.5}
        return validate_agent_output(result, RiskAgentOutput, "Risk Agent")


class FinancialSpecialistAgent:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.name = "Financial Agent"

    def analyze(self, text: str, filename: str) -> dict:
        raw = call_gemini(
            system="""You are a senior financial analyst specializing in contract financial risk.
Think step by step: (1) find all dollar amounts, (2) classify each, (3) determine direction,
(4) identify contingent exposures.
Respond ONLY with valid JSON. No markdown.""",
            user=f'''Extract all financial terms from this document.

DOCUMENT NAME: {filename}
DOCUMENT TEXT:
{text[:22000]}

Return ONLY this JSON:
{{
  "financial_terms": [
    {{
      "label": "Descriptive name",
      "value": "Exact dollar amount or formula from document",
      "type": "payment|penalty|liability|revenue|fee|indemnity|insurance|deposit|other",
      "direction": "payable|receivable|contingent|mutual",
      "notes": "Conditions or triggers",
      "confidence": 0.90
    }}
  ],
  "total_liability_exposure": "Maximum total financial exposure with dollar figure",
  "payment_schedule": "Exact description of payment timing",
  "penalty_clauses": ["Each penalty trigger and amount"],
  "confidence": 0.85
}}

Extract EVERY financial term. Use exact language from document.
For uncapped exposures state "Unlimited — no cap specified".''',
            max_tokens=2000,
            model_name=self.model_name,
        )
        result = parse_json(raw)
        if not result:
            result = {"financial_terms": [], "total_liability_exposure": "Could not determine", "payment_schedule": "Not found", "penalty_clauses": [], "confidence": 0.5}
        return validate_agent_output(result, FinancialAgentOutput, "Financial Agent")


class ObligationsSpecialistAgent:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.name = "Obligations Agent"

    def analyze(self, text: str, filename: str) -> dict:
        raw = call_gemini(
            system="""You are a contract compliance specialist.
An obligation is any requirement a party MUST perform, MUST NOT perform, or MUST ensure.
Think: (1) identify each party, (2) find every "shall", "must", "will", "agrees to",
(3) find deadline/timing, (4) find consequence of non-performance.
Every contract has obligations — empty list = error.
Respond ONLY with valid JSON. No markdown.""",
            user=f'''Extract ALL obligations from every party.

DOCUMENT NAME: {filename}
DOCUMENT TEXT:
{text[:22000]}

Return ONLY this JSON:
{{
  "obligations": [
    {{
      "party": "Exact name of party with obligation",
      "obligation": "Precise statement of what they must do",
      "deadline": "Exact timing or Ongoing",
      "consequences": "What happens if not fulfilled",
      "risk_level": "low|medium|high",
      "confidence": 0.90
    }}
  ],
  "key_dates": [
    {{
      "label": "Date name",
      "date": "Actual date or timeframe",
      "importance": "low|medium|high",
      "consequence": "What happens on this date"
    }}
  ],
  "termination_conditions": ["Complete description of each termination condition"],
  "confidence": 0.85
}}

CRITICAL: Return minimum 3-5 obligations. Look for: payment, delivery, reporting,
insurance, compliance, notification, confidentiality obligations.''',
            max_tokens=2000,
            model_name=self.model_name,
        )
        result = parse_json(raw)
        # Fallback if empty
        if not result.get("obligations"):
            logger.warning(f"Obligations agent empty for {filename}, retrying with simpler prompt")
            raw2 = call_gemini(
                system="You are a contract analyst. Extract obligations. Respond ONLY with valid JSON.",
                user=f'''List every obligation (things parties MUST do) in this contract.
Look for "shall", "must", "agrees to", "will provide".

{text[:15000]}

Return: {{"obligations":[{{"party":"...","obligation":"...","deadline":"...","consequences":"...","risk_level":"medium","confidence":0.8}}],"key_dates":[],"termination_conditions":[],"confidence":0.7}}''',
                max_tokens=1500,
                model_name=self.model_name,
            )
            result2 = parse_json(raw2)
            if result2.get("obligations"):
                return validate_agent_output(result2, ObligationsAgentOutput, "Obligations Agent")
        if not result:
            result = {"obligations": [], "key_dates": [], "termination_conditions": [], "confidence": 0.5}
        return validate_agent_output(result, ObligationsAgentOutput, "Obligations Agent")


class ComplianceSpecialistAgent:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.name = "Compliance Agent"

    def analyze(self, text: str, filename: str) -> dict:
        raw = call_gemini(
            system="""You are a compliance officer and contract auditor.
Check for all standard protective clauses: Force Majeure, Limitation of Liability,
Indemnification, Confidentiality, IP Ownership, Data Protection, Governing Law,
Dispute Resolution, Assignment Restrictions, Severability, Entire Agreement,
Amendment Process, Notice Requirements, Audit Rights, Insurance Requirements,
Non-Solicitation, Non-Compete.
Mark each as present or absent.
Respond ONLY with valid JSON. No markdown.""",
            user=f'''Audit this document for missing protections and compliance gaps.

DOCUMENT NAME: {filename}
DOCUMENT TEXT:
{text[:22000]}

Return ONLY this JSON:
{{
  "missing_clauses": [
    {{
      "clause": "Standard clause name",
      "importance": "required|recommended|optional",
      "risk_if_absent": "Specific risk created by absence",
      "confidence": 0.90
    }}
  ],
  "present_protections": ["Standard protective clauses that ARE present"],
  "compliance_notes": [
    {{
      "note": "Specific compliance observation",
      "severity": "low|medium|high"
    }}
  ],
  "governing_law": "Jurisdiction and state/country or Not specified",
  "dispute_resolution": "Arbitration/Litigation/Mediation/Not specified with details",
  "confidence": 0.85
}}''',
            max_tokens=1800,
            model_name=self.model_name,
        )
        result = parse_json(raw)
        if not result:
            result = {"missing_clauses": [], "present_protections": [], "compliance_notes": [], "governing_law": "Not specified", "dispute_resolution": "Not specified", "confidence": 0.5}
        return validate_agent_output(result, ComplianceAgentOutput, "Compliance Agent")


# ── Orchestrator ──────────────────────────────────────────────────────────────

class OrchestratorAgent:
    """
    Classifies document → runs 4 agents in parallel → synthesizes with final Gemini call.
    model_name is passed through to all specialists for easy model switching.
    """

    def __init__(self, model_name: str = "gemini-2.0-flash"):
        self.model_name = model_name
        self.specialists = {
            "risk":        RiskSpecialistAgent(model_name),
            "financial":   FinancialSpecialistAgent(model_name),
            "obligations": ObligationsSpecialistAgent(model_name),
            "compliance":  ComplianceSpecialistAgent(model_name),
        }

    def run(self, doc_text: str, filename: str) -> dict:
        trace = []
        pipeline_start = time.time()

        # Step 1 — Classify
        trace.append({"agent": "Orchestrator", "step": f"Received '{filename}'. Routing to 4 agents in parallel...", "status": "running"})

        routing_raw = call_gemini(
            system="Respond ONLY with valid JSON. No explanation.",
            user=f'''Classify this document.
Name: {filename}
Preview: {doc_text[:2000]}
Return: {{"document_type":"contract|nda|financial_report|sec_filing|amendment|term_sheet|other","primary_concern":"one sentence on top risk","estimated_complexity":"simple|moderate|complex"}}''',
            max_tokens=400,
            model_name=self.model_name,
        )
        routing = parse_json(routing_raw)
        doc_type = routing.get("document_type", "contract")
        complexity = routing.get("estimated_complexity", "moderate")

        trace.append({"agent": "Orchestrator", "step": f"Classified: {doc_type} ({complexity}). Launching all 4 agents simultaneously...", "status": "complete"})

        # Step 2 — Parallel execution
        results = {}
        errors = {}
        agent_times = {}

        def run_agent(key):
            t0 = time.time()
            try:
                result = self.specialists[key].analyze(doc_text, filename)
                return key, result, round(time.time() - t0, 1), None
            except Exception as e:
                logger.error(f"Agent {key} failed: {e}", exc_info=True)
                return key, None, round(time.time() - t0, 1), str(e)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(run_agent, key): key for key in self.specialists}
            for future in as_completed(futures):
                key, result, elapsed, err = future.result()
                agent_times[key] = elapsed
                if result:
                    results[key] = result
                else:
                    errors[key] = err or "Unknown error"

        # Defaults for failed agents
        defaults = {
            "risk":        {"executive_summary": "Agent failed.", "parties": [], "risk_flags": [], "confidence": 0.5},
            "financial":   {"financial_terms": [], "total_liability_exposure": "Unknown", "payment_schedule": "Not found", "penalty_clauses": [], "confidence": 0.5},
            "obligations": {"obligations": [], "key_dates": [], "termination_conditions": [], "confidence": 0.5},
            "compliance":  {"missing_clauses": [], "present_protections": [], "compliance_notes": [], "governing_law": "Not specified", "dispute_resolution": "Not specified", "confidence": 0.5},
        }
        for key in self.specialists:
            if key not in results:
                results[key] = defaults[key]

        # Trace each agent
        count_keys = {"risk": "risk_flags", "financial": "financial_terms", "obligations": "obligations", "compliance": "missing_clauses"}
        for key, agent in self.specialists.items():
            n = len(results[key].get(count_keys[key], []))
            conf = results[key].get("confidence", 0.85)
            elapsed = agent_times.get(key, 0)
            failed = key in errors
            trace.append({
                "agent": agent.name,
                "step": f"Failed ({errors.get(key)})." if failed else f"Complete — {n} item(s) at {conf:.0%} confidence. ({elapsed}s)",
                "status": "error" if failed else "complete",
            })

        # Step 3 — Deduplicate + score
        raw_flags = results["risk"].get("risk_flags", [])
        all_flags = _semantic_dedup(raw_flags)
        risk_score = self._compute_risk_score(all_flags, results)
        risk_lbl = self._risk_label(risk_score)
        overall_confidence = round(sum(results[k].get("confidence", 0.85) for k in results) / 4, 2)

        # Step 4 — Synthesis
        synthesis_summary = self._synthesize(filename, doc_type, risk_score, risk_lbl, all_flags, results, overall_confidence)

        total_elapsed = round(time.time() - pipeline_start, 1)
        sequential_estimate = sum(agent_times.values())

        trace.append({
            "agent": "Orchestrator",
            "step": (
                f"All agents complete. Risk Score: {risk_score}/100 ({risk_lbl}). "
                f"Confidence: {overall_confidence:.0%}. "
                f"Parallel: {total_elapsed}s (vs ~{sequential_estimate:.0f}s sequential — "
                f"{sequential_estimate/max(total_elapsed,1):.1f}x faster)."
            ),
            "status": "complete",
        })
        trace.append({"agent": "Orchestrator", "step": synthesis_summary, "status": "complete"})

        return {
            "analysis": {
                "document_type": doc_type,
                "filename": filename,
                "risk_score": risk_score,
                "risk_label": risk_lbl,
                "overall_confidence": overall_confidence,
                "executive_summary": synthesis_summary,
                "parties": results["risk"].get("parties", []),
                "risk_flags": all_flags,
                "financial_terms": results["financial"].get("financial_terms", []),
                "total_liability_exposure": results["financial"].get("total_liability_exposure", "Unknown"),
                "penalty_clauses": results["financial"].get("penalty_clauses", []),
                "payment_schedule": results["financial"].get("payment_schedule", ""),
                "obligations": results["obligations"].get("obligations", []),
                "key_dates": results["obligations"].get("key_dates", []),
                "termination_conditions": results["obligations"].get("termination_conditions", []),
                "missing_clauses": results["compliance"].get("missing_clauses", []),
                "present_protections": results["compliance"].get("present_protections", []),
                "compliance_notes": results["compliance"].get("compliance_notes", []),
                "governing_law": results["compliance"].get("governing_law", "Not specified"),
                "dispute_resolution": results["compliance"].get("dispute_resolution", "Not specified"),
                "risk_by_category": self._risk_by_category(all_flags),
                "severity_distribution": self._severity_distribution(all_flags),
                "confidence_scores": {k: results[k].get("confidence", 0.85) for k in results},
                "pipeline_time_seconds": total_elapsed,
                "sequential_estimate_seconds": sequential_estimate,
                "model": self.model_name,
            },
            "trace": trace,
        }

    def _synthesize(self, filename, doc_type, risk_score, risk_lbl, flags, results, confidence) -> str:
        flag_summary = "\n".join(
            f"- [{f.get('severity','?').upper()}] {f.get('title','')}: {f.get('description','')}"
            for f in flags[:6]
        )
        missing = [c.get("clause", "") for c in results["compliance"].get("missing_clauses", [])[:4]]
        exposure = results["financial"].get("total_liability_exposure", "Unknown")
        obligations_count = len(results["obligations"].get("obligations", []))

        try:
            return call_gemini(
                system="You are a senior investment banker. Write concise, precise, accountable executive summaries.",
                user=f"""You are a Managing Director writing a one-paragraph board-level summary.

Document: {filename} ({doc_type})
Risk Score: {risk_score}/100 — {risk_lbl}
Confidence: {confidence:.0%}

Key Risk Flags:
{flag_summary or "No critical flags identified."}

Financial Exposure: {exposure}
Missing Protections: {', '.join(missing) if missing else 'None identified'}
Obligations Found: {obligations_count}

Write 3-4 sentences: document type/parties/rating, 1-2 critical specific risks with section refs,
financial exposure, clear action recommendation (sign/negotiate/reject).
Be specific, cite findings, no generic language.""",
                max_tokens=400,
                model_name=self.model_name,
            ).strip()
        except Exception as e:
            logger.warning(f"Synthesis failed: {e}")
            return results["risk"].get("executive_summary", f"Risk Score {risk_score}/100 ({risk_lbl}).")

    def _compute_risk_score(self, flags: list, results: dict) -> int:
        if not flags:
            missing = results["compliance"].get("missing_clauses", [])
            required_missing = sum(1 for c in missing if c.get("importance") == "required")
            return min(35, 5 + required_missing * 5)
        weights = {"critical": 100, "high": 70, "medium": 38, "low": 12}
        total_weight = 0
        weighted_sum = 0
        for f in flags:
            conf = float(f.get("confidence", 0.8))
            w = weights.get(f.get("severity", "low"), 12) * conf
            weighted_sum += w
            total_weight += conf
        base_score = int(weighted_sum / max(total_weight, 1))
        critical_count = sum(1 for f in flags if f.get("severity") == "critical")
        boost = min(25, critical_count * 10)
        missing = results["compliance"].get("missing_clauses", [])
        penalty = min(10, sum(1 for c in missing if c.get("importance") == "required") * 3)
        return min(100, max(5, base_score + boost + penalty))

    def _risk_label(self, score: int) -> str:
        if score >= 75: return "Critical"
        if score >= 50: return "High"
        if score >= 28: return "Medium"
        return "Low"

    def _risk_by_category(self, flags: list) -> dict:
        cats = {"Legal": 0, "Financial": 0, "Operational": 0, "Compliance": 0}
        for f in flags:
            cat = f.get("category", "Legal")
            cats[cat] = cats.get(cat, 0) + 1 if cat in cats else cats.setdefault(cat, 1)
        return cats

    def _severity_distribution(self, flags: list) -> dict:
        dist = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for f in flags:
            sev = f.get("severity", "low").lower()
            if sev in dist:
                dist[sev] += 1
        return dist