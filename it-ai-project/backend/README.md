# IT-AI backend (FastAPI)

```bash
cd it-ai-backend
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # add your ANTHROPIC_API_KEY (works without a key using rule fallback)
uvicorn main:app --reload   # http://localhost:8000/docs
```

## Connect the existing it-ai.html frontend
In `submitTicket()` replace the `const a=analyze(ti,de), id=...; T.unshift(...)` block with:

```js
fetch('http://localhost:8000/tickets',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({title:ti,description:de})})
 .then(r=>r.json()).then(t=>{ location.hash='#/analysis/'+t.id; /* store t in T and map fields:
   a = {cat:t.category, sub:t.subcategory, pri:t.priority, sent:t.sentiment, conf:t.confidence,
        reason:t.reasoning, rel:t.kb_relevance, kb:{t:t.kb_title,s:t.solution}, auto:!!t.can_auto_resolve,
        team:t.team, why:t.escalation_reason} */ });
```
Endpoints: POST /tickets, GET /tickets, GET /tickets/{id}, POST /tickets/{id}/confirm {solved},
/escalate, /resolve, GET /kb, POST /chat, GET /stats.

## Production upgrades
- Replace TF-IDF in `retrieve()` with embeddings + pgvector (or Pinecone/Chroma).
- Add Google OAuth / email login (e.g. `fastapi-users`) and restrict CORS origins.
- Safety rule lives in `analyze()`: CRITICAL, Security, Infrastructure and Database never auto-resolve.
