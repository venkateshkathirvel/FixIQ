
import os, json, re
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime, JSON
from sqlalchemy.orm import declarative_base, sessionmaker
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from kb import KB

load_dotenv()
THRESHOLD = float(os.getenv("AUTO_RESOLVE_THRESHOLD", 85))
HIGH_RISK = {"Infrastructure", "Security", "Database"}   # humans must approve these
TEAMS = {"Network": "Network Team", "Infrastructure": "DevOps / Infrastructure", "Security": "Security Team",
         "Database": "Database Admins", "Email": "Messaging Team", "Hardware": "Desktop Support",
         "Software": "Desktop Support", "Account & Access": "Service Desk"}

# ---------- database ----------
engine = create_engine(os.getenv("DATABASE_URL", "sqlite:///./itai.db"))
Session = sessionmaker(engine)
Base = declarative_base()

class Ticket(Base):
    __tablename__ = "tickets"
    id = Column(Integer, primary_key=True)
    title = Column(String(200)); description = Column(Text)
    category = Column(String(50)); subcategory = Column(String(80))
    priority = Column(String(10)); sentiment = Column(String(20))
    confidence = Column(Float); reasoning = Column(Text)
    kb_title = Column(String(120)); kb_relevance = Column(Float)
    solution = Column(JSON); team = Column(String(60))
    can_auto_resolve = Column(Integer, default=0); escalation_reason = Column(Text)
    status = Column(String(20), default="OPEN"); method = Column(String(20), default="")
    created_at = Column(DateTime, default=datetime.utcnow); resolved_at = Column(DateTime, nullable=True)
    history = Column(JSON, default=list)

Base.metadata.create_all(engine)

# ---------- RAG (TF-IDF now; swap for embeddings + pgvector in production) ----------
_docs = [d["title"] + " " + d["category"] + " " + d["text"] for d in KB]
_vec = TfidfVectorizer(stop_words="english").fit(_docs)
_mat = _vec.transform(_docs)

def retrieve(query: str):
    sims = cosine_similarity(_vec.transform([query]), _mat)[0]
    i = int(sims.argmax())
    return (KB[i], round(float(min(sims[i] * 2.2, 0.99)) * 100)) if sims[i] > 0.05 else (None, 0)

# ---------- LLM classification ----------
SYSTEM = """You are an IT service-desk triage engine. Return ONLY JSON:
{"category": one of [Network,Hardware,Software,Account & Access,Email,Security,VPN,Infrastructure,Database,Application,Other],
 "subcategory": str, "priority": "LOW|MEDIUM|HIGH|CRITICAL", "sentiment": "Neutral|Frustrated|Urgent",
 "confidence": 0-100, "reasoning": "one or two sentences"}
Priority: production/security impact = CRITICAL; many users or hard deadline = HIGH; one user blocked = MEDIUM; peripheral = LOW."""

def classify(text: str) -> dict:
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            import anthropic
            msg = anthropic.Anthropic().messages.create(
                model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5"), max_tokens=400,
                system=SYSTEM, messages=[{"role": "user", "content": text}])
            raw = re.search(r"\{.*\}", msg.content[0].text, re.S).group(0)
            return json.loads(raw)
        except Exception as e:
            print("LLM failed, using fallback:", e)
    return rule_fallback(text)

def rule_fallback(t: str) -> dict:
    t = t.lower()
    rules = [(r"production|server.*down|outage|customers", "Infrastructure", "CRITICAL", 91),
             (r"suspicious|phish|malware", "Security", "CRITICAL", 93),
             (r"database|sql", "Database", "HIGH", 78), (r"vpn", "Network", "HIGH", 96),
             (r"password|forgot|locked", "Account & Access", "MEDIUM", 94), (r"email|outlook", "Email", "MEDIUM", 90),
             (r"wi-?fi", "Network", "MEDIUM", 92), (r"mouse|keyboard|monitor|printer", "Hardware", "LOW", 92)]
    for rx, cat, pri, conf in rules:
        if re.search(rx, t):
            return {"category": cat, "subcategory": cat, "priority": pri, "sentiment": "Frustrated",
                    "confidence": conf, "reasoning": f"Keyword match for {cat}."}
    return {"category": "Other", "subcategory": "General", "priority": "MEDIUM", "sentiment": "Neutral",
            "confidence": 60, "reasoning": "No clear match."}

def analyze(title: str, desc: str) -> dict:
    text = f"{title}\n{desc}"
    c = classify(text)
    doc, rel = retrieve(text)
    # blend model confidence with retrieval strength
    conf = round(c["confidence"] * 0.8 + (rel if doc else 0) * 0.2) if doc else min(c["confidence"], 70)
    pri, cat = c["priority"], c["category"]
    if cat == "VPN": cat = "Network"
    auto = conf >= THRESHOLD and pri != "CRITICAL" and cat not in HIGH_RISK and doc is not None
    if auto: why = f"Confidence {conf}% is above the {THRESHOLD:.0f}% threshold and the issue is low-risk."
    elif pri == "CRITICAL": why = "Critical tickets always require human review. " + c["reasoning"]
    elif cat in HIGH_RISK: why = f"{cat} issues need human approval before any change."
    else: why = f"Confidence {conf}% is below the {THRESHOLD:.0f}% threshold."
    return {"category": cat, "subcategory": c["subcategory"], "priority": pri, "sentiment": c["sentiment"],
            "confidence": conf, "reasoning": c["reasoning"], "kb_title": doc["title"] if doc else None,
            "kb_relevance": rel, "solution": doc["steps"] if doc else [], "team": TEAMS.get(cat, "Service Desk"),
            "can_auto_resolve": int(auto), "escalation_reason": "" if auto else why, "why": why}

# ---------- API ----------
app = FastAPI(title="IT-AI")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class NewTicket(BaseModel):
    title: str; description: str = ""
class Confirm(BaseModel):
    solved: bool
class Chat(BaseModel):
    message: str

def out(t: Ticket) -> dict:
    d = {c.name: getattr(t, c.name) for c in t.__table__.columns}
    d["created_at"] = t.created_at.isoformat(); d["resolved_at"] = t.resolved_at.isoformat() if t.resolved_at else None
    return d

def note(t, msg): t.history = (t.history or []) + [{"at": datetime.utcnow().isoformat(), "event": msg}]

def load(s, tid):
    t = s.get(Ticket, tid)
    if not t: raise HTTPException(404, "Ticket not found")
    return t

@app.post("/tickets")
def create(body: NewTicket):
    a = analyze(body.title, body.description)
    with Session() as s:
        t = Ticket(title=body.title, description=body.description, status="OPEN",
                   **{k: a[k] for k in ["category","subcategory","priority","sentiment","confidence","reasoning",
                                        "kb_title","kb_relevance","solution","team","can_auto_resolve","escalation_reason"]})
        t.history = []
        for m in ["Ticket created", "AI classification completed", f"Knowledge retrieved: {a['kb_title']}", "AI solution generated"]: note(t, m)
        if not a["can_auto_resolve"]: t.status = "NEEDS_REVIEW"
        s.add(t); s.commit(); s.refresh(t); return out(t)

@app.get("/tickets")
def list_tickets():
    with Session() as s: return [out(t) for t in s.query(Ticket).order_by(Ticket.id.desc()).all()]

@app.get("/tickets/{tid}")
def get_ticket(tid: int):
    with Session() as s: return out(load(s, tid))

@app.post("/tickets/{tid}/confirm")
def confirm(tid: int, body: Confirm):
    with Session() as s:
        t = load(s, tid)
        if body.solved and t.can_auto_resolve:
            t.status, t.method, t.resolved_at = "RESOLVED", "AI Assisted", datetime.utcnow()
            note(t, "Employee confirmed solution"); note(t, "Ticket resolved")
        else:
            t.status, t.method = "ESCALATED", "Human"; note(t, f"Escalated to {t.team}")
        s.commit(); return out(t)

@app.post("/tickets/{tid}/escalate")
def escalate(tid: int):
    with Session() as s:
        t = load(s, tid); t.status, t.method = "ESCALATED", "Human"
        note(t, f"Escalated to {t.team}: {t.escalation_reason or 'requested by user'}"); s.commit(); return out(t)

@app.post("/tickets/{tid}/resolve")   # engineer action
def resolve(tid: int):
    with Session() as s:
        t = load(s, tid); t.status, t.method, t.resolved_at = "RESOLVED", "Human", datetime.utcnow()
        note(t, "Resolved by engineer"); s.commit(); return out(t)

@app.get("/kb")
def kb(): return [{"title": d["title"], "category": d["category"], "steps": d["steps"]} for d in KB]

@app.post("/chat")
def chat(body: Chat):
    a = analyze(body.message, "")
    return {"reply_meta": {k: a[k] for k in ["category","subcategory","priority","confidence","team"]},
            "steps": a["solution"] if a["can_auto_resolve"] else [], "needs_human": not a["can_auto_resolve"], "why": a["why"]}

@app.get("/stats")
def stats():
    with Session() as s:
        ts = s.query(Ticket).all(); n = len(ts) or 1
        ai = sum(t.method == "AI Assisted" for t in ts); hu = sum(t.method == "Human" and t.status == "RESOLVED" for t in ts)
        by = {}
        for t in ts: by[t.category] = by.get(t.category, 0) + 1
        return {"total": len(ts), "ai_resolved": ai, "human_resolved": hu,
                "escalated": sum(t.status == "ESCALATED" for t in ts), "ai_rate": round(ai / n * 100), "by_category": by}
