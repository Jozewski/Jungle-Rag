# 🌿 The Survival Guide — Setup

A RAG-powered survival knowledge base covering **Jungle, Desert, Alaska/Tundra, and Ocean/Shipwreck** survival.

---

## Local Setup (5 minutes)

### 1. Get a free Groq API key
Go to **https://console.groq.com** → Sign up → Create API key (free, no credit card).

### 2. Install dependencies
```bash
cd jungle-rag
pip install -r requirements.txt
```

### 3. Set your API key
**Mac/Linux:**
```bash
export GROQ_API_KEY=your_key_here
```
**Windows (PowerShell):**
```powershell
$env:GROQ_API_KEY="your_key_here"
```

### 4. Run the server
```bash
uvicorn main:app --reload --port 8000
```

### 5. Open the app
Go to **http://localhost:8000** in your browser.

On first launch, the app will build the vector index (takes ~30–60 seconds to download the embedding model). Subsequent launches are instant.

---

## Deployment on Railway

### 1. Install Railway CLI
```bash
npm install -g @railway/cli
railway login
```

### 2. Deploy
```bash
cd jungle-rag
railway init
railway up
```

### 3. Set your environment variable in Railway dashboard
Go to your project → Variables → Add:
```
GROQ_API_KEY = your_key_here
```

### 4. Access your deployed app
Railway provides a public URL like `https://jungle-rag-production.up.railway.app`

---

## How It Works (RAG Pipeline)

```
User Question
     │
     ▼
[Embedding Model]  ←──  sentence-transformers/all-MiniLM-L6-v2 (local, free)
     │
     ▼
[ChromaDB Vector Search]  ←──  top-5 most relevant text chunks
     │
     ▼
[Grounded Prompt]  ←──  chunks injected into context, strict "only use context" rule
     │
     ▼
[Groq LLM]  ←──  llama-3.1-8b-instant (free tier, very fast)
     │
     ▼
Answer + Sources displayed in UI
```

---

## Knowledge Base

| File | Topics |
|------|--------|
| `shelter.md` | Lean-to, A-frame, thatching, jungle shelter |
| `food_and_water.md` | Finding water, edible plants, insects, purification |
| `navigation.txt` | Sun/star navigation, waterways, terrain, marking trail |
| `wildlife_dangers.md` | Snakes, insects, predators, leeches |
| `fire_making.md` | Bow drill, flint, fire structures |
| `first_aid.json` | Wounds, snakebite, dehydration, hypothermia, fever |
| `signaling_rescue.txt` | Flares, mirrors, ground signals, EPIRB |
| `desert_survival.md` | Heat, water collection, sun shelter, desert navigation |
| `alaska_tundra_survival.md` | Hypothermia, snow shelter, frostbite, ice travel |
| `ocean_shipwreck_survival.md` | Life raft, water at sea, signaling, fishing |

---

## Adding to the Knowledge Base

Drop any `.md`, `.txt`, or `.json` file into the `knowledge_base/` folder and restart the server with `--force-rebuild`:

```bash
REBUILD_INDEX=true uvicorn main:app --reload --port 8000
```

Or add to `main.py`'s lifespan: `rag.build_index(force_rebuild=True)`.
