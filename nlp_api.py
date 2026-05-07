"""
nlp_api.py  (v2 — multilingual + SQLite)
==========================================
FastAPI backend for the Ooredoo multilingual complaint platform.

Endpoints:
  GET  /form                   — multilingual complaint submission form
  POST /api/complaints/submit  — submit + analyze + store in SQLite
  POST /api/complaints/analyze — analyze text without storing
  GET  /api/complaints         — list complaints with filters
  GET  /api/complaints/stats   — aggregated stats
  GET  /api/complaints/{id}    — get one complaint
  PUT  /api/complaints/{id}/status  — update status

Run:
    uvicorn src.nlp.nlp_api:app --reload --port 8000
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.nlp.multilingual_nlp_pipeline import MultilingualNLPPipeline
from src.nlp.complaint_db import ComplaintDB

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Ooredoo NLP Complaint Platform",
    description="Multilingual complaint analysis — Arabic, French, English",
    version="2.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

_pipe = MultilingualNLPPipeline()
_db   = ComplaintDB()


# ── Pydantic models ────────────────────────────────────────────────────────────
class ComplaintSubmit(BaseModel):
    text:    str           = Field(..., min_length=5, max_length=3000)
    msisdn:  Optional[str] = None
    city:    Optional[str] = None
    segment: Optional[str] = None
    channel: Optional[str] = "web"


class StatusUpdate(BaseModel):
    status: str = Field(..., pattern="^(open|in_progress|resolved)$")


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    return """<html><body style="font-family:Arial;max-width:500px;margin:40px auto">
    <h2>📡 Ooredoo NLP API v2</h2>
    <ul>
      <li><a href="/form">Complaint Form (AR/FR/EN)</a></li>
      <li><a href="/docs">API Documentation</a></li>
      <li><a href="/api/complaints/stats">Live Stats</a></li>
    </ul>
    </body></html>"""


@app.post("/api/complaints/submit", tags=["Complaints"])
async def submit_complaint(c: ComplaintSubmit):
    """Submit, analyze, and store a complaint in SQLite."""
    nlp    = _pipe.analyze(c.text)
    cid    = _db._generate_id()

    record = {
        "complaint_id":   cid,
        "submitted_at":   datetime.now().isoformat(),
        "msisdn":         c.msisdn,
        "city_input":     c.city,
        "segment":        c.segment,
        "channel":        c.channel or "web",
        "text_original":  c.text,
        **nlp,
    }
    _db.insert(record)

    resp_hours = {"très urgent": 2, "urgent": 8, "normal": 24}.get(
        nlp["urgency_level"], 24
    )
    lang_label = {"ar": "العربية", "fr": "Français", "en": "English"}.get(
        nlp["language"], nlp["language"]
    )

    return {
        "complaint_id":            cid,
        "language_detected":       lang_label,
        "category":                nlp["category"],
        "sentiment":               nlp["sentiment"],
        "urgency_level":           nlp["urgency_level"],
        "urgency_score":           nlp["urgency_score"],
        "city_detected":           nlp["city"],
        "estimated_response_hours": resp_hours,
        "message": (
            f"Complaint registered (ID: {cid}). "
            f"Our team will contact you within {resp_hours}h."
        ),
    }


@app.post("/api/complaints/analyze", tags=["NLP"])
async def analyze_only(c: ComplaintSubmit):
    """Analyze text without storing — useful for live demo."""
    return _pipe.analyze(c.text)


@app.get("/api/complaints/stats", tags=["Analytics"])
async def get_stats():
    """Aggregated statistics from SQLite."""
    return _db.stats()


@app.get("/api/complaints", tags=["Complaints"])
async def list_complaints(
    language:  Optional[str] = Query(None, example="ar"),
    urgency:   Optional[str] = Query(None, example="urgent"),
    sentiment: Optional[str] = Query(None, example="critique"),
    status:    Optional[str] = Query(None, example="open"),
    limit:     int            = Query(100, le=500),
):
    """List complaints with optional filters."""
    df = _db.to_dataframe(
        language=language, urgency=urgency,
        sentiment=sentiment, status=status, limit=limit
    )
    if df.empty:
        return {"total": 0, "complaints": []}
    records = df.to_dict(orient="records")
    return {"total": len(records), "complaints": records}


@app.get("/api/complaints/{complaint_id}", tags=["Complaints"])
async def get_complaint(complaint_id: str):
    """Get a single complaint by ID."""
    df = _db.to_dataframe(limit=10000)
    row = df[df["complaint_id"] == complaint_id]
    if row.empty:
        raise HTTPException(404, f"Complaint {complaint_id} not found")
    return row.iloc[0].to_dict()


@app.put("/api/complaints/{complaint_id}/status", tags=["Complaints"])
async def update_status(complaint_id: str, body: StatusUpdate):
    """Update complaint status: open → in_progress → resolved."""
    _db.update_status(complaint_id, body.status)
    return {"complaint_id": complaint_id, "status": body.status}
@app.delete("/api/complaints/{complaint_id}", tags=["Complaints"])
async def delete_complaint(complaint_id: str):
    """Delete a complaint from the database."""
    with _db._conn() as conn:
        cursor = conn.execute(
            "DELETE FROM complaints WHERE complaint_id = ?",
            (complaint_id,)
        )
        if cursor.rowcount == 0:
            raise HTTPException(404, f"Complaint {complaint_id} not found")
    return {"complaint_id": complaint_id, "deleted": True}

# ── Multilingual complaint form ────────────────────────────────────────────────
@app.get("/form", response_class=HTMLResponse, tags=["Customer Portal"])
async def complaint_form():
    return """<!DOCTYPE html>
<html lang="ar" dir="auto">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ooredoo — Complaint / Réclamation / شكوى</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Tahoma,sans-serif;
     background:#f0f2f5;min-height:100vh;display:flex;align-items:center;
     justify-content:center;padding:20px}
.card{background:white;border-radius:16px;box-shadow:0 4px 32px rgba(0,0,0,.1);
      padding:36px;width:100%;max-width:580px}
.header{text-align:center;margin-bottom:28px}
.logo{width:52px;height:52px;background:#e30613;border-radius:12px;
      display:inline-flex;align-items:center;justify-content:center;
      color:white;font-size:26px;font-weight:700;margin-bottom:10px}
h1{font-size:20px;color:#1a1a2e;margin-bottom:4px}
.subtitle{font-size:13px;color:#666}
.lang-tabs{display:flex;gap:8px;margin-bottom:24px;
           border-bottom:2px solid #f0f2f5;padding-bottom:12px}
.lang-tab{padding:8px 18px;border:1.5px solid #e5e7eb;border-radius:20px;
          cursor:pointer;font-size:13px;font-weight:600;color:#666;
          background:white;transition:.2s}
.lang-tab.active{background:#e30613;border-color:#e30613;color:white}
.field{margin-bottom:18px}
label{display:block;font-size:13px;font-weight:600;color:#374151;
      margin-bottom:5px}
input,select,textarea{width:100%;padding:11px 14px;
      border:1.5px solid #e5e7eb;border-radius:8px;font-size:14px;
      color:#111;outline:none;transition:.2s;font-family:inherit}
input:focus,select:focus,textarea:focus{border-color:#e30613}
textarea{resize:vertical;min-height:130px;line-height:1.6}
.row{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.btn{width:100%;padding:14px;background:#e30613;color:white;border:none;
     border-radius:8px;font-size:15px;font-weight:700;cursor:pointer;
     margin-top:6px;transition:.2s;font-family:inherit}
.btn:hover{background:#c0000f}
.btn:disabled{background:#ccc;cursor:default}
.result{display:none;margin-top:20px;padding:18px;border-radius:10px;
        border:1.5px solid}
.result.ok{background:#f0fdf4;border-color:#16a34a}
.result.err{background:#fef2f2;border-color:#dc2626}
.badge{display:inline-block;padding:3px 10px;border-radius:16px;
       font-size:11px;font-weight:700;margin:2px}
.b-red{background:#fee2e2;color:#991b1b}
.b-orange{background:#fef3c7;color:#92400e}
.b-green{background:#d1fae5;color:#065f46}
.b-purple{background:#ede9fe;color:#5b21b6}
.hint{font-size:11px;color:#9ca3af;margin-top:3px}
.lang-hint{font-size:12px;color:#6b7280;text-align:center;
           margin-bottom:16px;padding:8px;background:#f9fafb;
           border-radius:8px}
</style>
</head>
<body>
<div class="card">
  <div class="header">
    <div class="logo">O</div>
    <h1>Ooredoo</h1>
    <p class="subtitle">
      Complaint &nbsp;|&nbsp; Réclamation &nbsp;|&nbsp; شكوى
    </p>
  </div>

  <div class="lang-tabs">
    <button class="lang-tab active" onclick="setLang('fr')">🇫🇷 Français</button>
    <button class="lang-tab"       onclick="setLang('ar')">🇹🇳 عربي</button>
    <button class="lang-tab"       onclick="setLang('en')">🇬🇧 English</button>
  </div>

  <div class="lang-hint" id="lang-hint">
    Écrivez en français, arabe, ou anglais — la langue sera détectée automatiquement.
  </div>

  <div class="row">
    <div class="field">
      <label id="lbl-msisdn">Numéro MSISDN</label>
      <input type="text" id="msisdn" placeholder="ex: 21612345678">
    </div>
    <div class="field">
      <label id="lbl-city">Ville / City / مدينة</label>
      <input type="text" id="city" placeholder="ex: Tunis, Sfax, تونس...">
    </div>
  </div>

  <div class="row">
    <div class="field">
      <label id="lbl-segment">Segment</label>
      <select id="segment">
        <option value="">--</option>
        <option>Standard</option>
        <option>Premium</option>
        <option>Enterprise</option>
        <option>VIP</option>
      </select>
    </div>
    <div class="field">
      <label id="lbl-channel">Canal / Channel / القناة</label>
      <select id="channel">
        <option value="web">Portail web / Web portal / البوابة</option>
        <option value="app">App mobile / تطبيق</option>
        <option value="social">Réseaux sociaux / Social / اجتماعي</option>
      </select>
    </div>
  </div>

  <div class="field">
    <label id="lbl-text">Décrivez votre problème *</label>
    <textarea id="complaint_text"
      placeholder="Ex FR: Mon réseau 4G coupe à Sfax depuis 3 jours...
Ex AR: شبكتي مقطوعة في تونس منذ 3 أيام...
Ex EN: My 4G keeps dropping in Tunis since yesterday..."></textarea>
    <p class="hint" id="txt-hint">
      Minimum 10 characters — language detected automatically
    </p>
  </div>

  <button class="btn" id="submitBtn" onclick="doSubmit()">
    <span id="btn-text">Soumettre</span>
  </button>

  <div class="result" id="result"></div>
</div>

<script>
const LABELS = {
  fr: {
    msisdn:  "Numéro MSISDN",
    city:    "Ville",
    segment: "Segment",
    channel: "Canal",
    text:    "Décrivez votre problème *",
    hint:    "Minimum 10 caractères — langue détectée automatiquement",
    btn:     "Soumettre la réclamation",
    hint2:   "Écrivez en français, arabe, ou anglais — la langue sera détectée automatiquement.",
  },
  ar: {
    msisdn:  "رقم الهاتف",
    city:    "المدينة",
    segment: "الشريحة",
    channel: "القناة",
    text:    "اشرح مشكلتك *",
    hint:    "10 أحرف على الأقل — اللغة تُكشف تلقائياً",
    btn:     "إرسال الشكوى",
    hint2:   "يمكنك الكتابة بالعربية أو الفرنسية أو الإنجليزية — اللغة تُكشف تلقائياً.",
  },
  en: {
    msisdn:  "MSISDN Number",
    city:    "City",
    segment: "Segment",
    channel: "Channel",
    text:    "Describe your problem *",
    hint:    "Minimum 10 characters — language auto-detected",
    btn:     "Submit complaint",
    hint2:   "Write in French, Arabic, or English — language is detected automatically.",
  },
};

function setLang(lang) {
  document.querySelectorAll('.lang-tab').forEach(t => t.classList.remove('active'));
  event.target.classList.add('active');
  const L = LABELS[lang];
  document.getElementById('lbl-msisdn').textContent  = L.msisdn;
  document.getElementById('lbl-city').textContent    = L.city;
  document.getElementById('lbl-segment').textContent = L.segment;
  document.getElementById('lbl-channel').textContent = L.channel;
  document.getElementById('lbl-text').textContent    = L.text;
  document.getElementById('txt-hint').textContent    = L.hint;
  document.getElementById('btn-text').textContent    = L.btn;
  document.getElementById('lang-hint').textContent   = L.hint2;
  document.documentElement.dir = lang === 'ar' ? 'rtl' : 'ltr';
}

async function doSubmit() {
  const text = document.getElementById('complaint_text').value.trim();
  if (text.length < 10) {
    alert('Please write at least 10 characters / Minimum 10 caractères / 10 أحرف على الأقل');
    return;
  }
  const btn = document.getElementById('submitBtn');
  btn.disabled = true;
  document.getElementById('btn-text').textContent = '⏳ Analyzing...';

  const payload = {
    text:    text,
    msisdn:  document.getElementById('msisdn').value  || null,
    city:    document.getElementById('city').value    || null,
    segment: document.getElementById('segment').value || null,
    channel: document.getElementById('channel').value,
  };

  try {
    const res  = await fetch('/api/complaints/submit', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    const uc   = data.urgency_level === 'très urgent' ? 'b-red'
               : data.urgency_level === 'urgent'      ? 'b-orange' : 'b-green';

    document.getElementById('result').style.display = 'block';
    document.getElementById('result').className     = 'result ok';
    document.getElementById('result').innerHTML = `
      <strong>${data.complaint_id}</strong> — ${data.message}<br><br>
      <span class="badge ${uc}">${data.urgency_level.toUpperCase()}</span>
      <span class="badge b-purple">${data.category}</span>
      <span class="badge" style="background:#e0f2fe;color:#0369a1">${data.language_detected}</span>
      <span class="badge" style="background:#f3f4f6;color:#374151">${data.sentiment}</span>
      ${data.city_detected ? `<span class="badge" style="background:#fef9c3;color:#713f12">${data.city_detected}</span>` : ''}
    `;
    document.getElementById('complaint_text').value = '';
  } catch(e) {
    document.getElementById('result').style.display = 'block';
    document.getElementById('result').className     = 'result err';
    document.getElementById('result').innerHTML     = '<strong>Error</strong> — Please try again.';
  }
  btn.disabled = false;
  document.getElementById('btn-text').textContent = 'Soumettre';
}
</script>
</body>
</html>"""