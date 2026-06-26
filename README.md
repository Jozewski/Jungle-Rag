# 🌿 The Survival Guide

A RAG-powered (Retrieval Augmented Generation) survival knowledge base API that provides expert wilderness and emergency survival guidance across multiple environments: **Jungle, Desert, Alaska/Tundra, and Ocean/Shipwreck**.

Built with FastAPI, ChromaDB, and Groq's free Llama 3.1 inference.

---

## Features

- **Grounded AI Responses**: Uses RAG to ensure answers are based on actual survival knowledge, not hallucinations
- **Multi-Environment Coverage**: Comprehensive guides for jungle, desert, arctic, and ocean survival
- **Fast & Free**: Powered by Groq's free tier (llama-3.1-8b-instant)
- **Source Citations**: Every answer includes relevant source chunks with similarity scores
- **Simple Web UI**: Clean interface for asking survival questions
- **REST API**: Easy integration with `/api/query` endpoint

---

## Tech Stack

- **Backend**: FastAPI
- **Vector Database**: ChromaDB
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **LLM**: Groq (Llama 3.1 8B Instant)
- **Deployment**: Railway-ready

---

## Quick Start

### Prerequisites

- Python 3.8+
- Free Groq API key from [console.groq.com](https://console.groq.com)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Jozewski/Jungle-Rag.git
cd jungle-rag
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file and add your Groq API key:
```bash
GROQ_API_KEY=your_key_here
```

4. Run the server:
```bash
uvicorn main:app --reload --port 8000
```

5. Open your browser to **http://localhost:8000**

On first launch, the app builds the vector index (takes ~30–60 seconds to download the embedding model). Subsequent launches are instant.

---

## How It Works (RAG Pipeline)

```
User Question
     │
     ▼
[Embedding Model]  ←  sentence-transformers/all-MiniLM-L6-v2 (local, free)
     │
     ▼
[ChromaDB Vector Search]  ←  top-5 most relevant text chunks
     │
     ▼
[Grounded Prompt]  ←  chunks injected into context with strict "only use context" rule
     │
     ▼
[Groq LLM]  ←  llama-3.1-8b-instant (free tier, fast)
     │
     ▼
Answer + Sources displayed in UI
```

---

## Knowledge Base

The `knowledge_base/` directory contains comprehensive survival guides:

| File | Topics |
|------|--------|
| `shelter.md` | Lean-to, A-frame, thatching, jungle shelter |
| `food_and_water.md` | Finding water, edible plants, insects, purification |
| `navigation.txt` | Sun/star navigation, waterways, terrain, trail marking |
| `wildlife_dangers.md` | Snakes, insects, predators, leeches |
| `fire_making.md` | Bow drill, flint, fire structures |
| `first_aid.json` | Wounds, snakebite, dehydration, hypothermia, fever |
| `signaling_rescue.txt` | Flares, mirrors, ground signals, EPIRB |
| `desert_survival.md` | Heat management, water collection, sun shelter, navigation |
| `alaska_tundra_survival.md` | Hypothermia, snow shelters, frostbite, ice travel |
| `ocean_shipwreck_survival.md` | Life raft usage, water at sea, signaling, fishing |

---

## API Endpoints

### `POST /api/query`

Ask a survival question and get a grounded answer with sources.

**Request:**
```json
{
  "question": "How do I find water in the jungle?"
}
```

**Response:**
```json
{
  "answer": "According to the food and water guide...",
  "sources": [
    {
      "source": "Food And Water",
      "text": "In jungle environments, look for...",
      "score": 0.85
    }
  ],
  "question": "How do I find water in the jungle?"
}
```

### `GET /api/health`

Health check endpoint.

### `GET /api/topics`

List all available knowledge base topics.

---

## Deployment on Railway

1. Install Railway CLI:
```bash
npm install -g @railway/cli
railway login
```

2. Deploy:
```bash
railway init
railway up
```

3. Set environment variable in Railway dashboard:
```
GROQ_API_KEY = your_key_here
```

4. Access your deployed app at the Railway-provided URL

---

## Adding Custom Knowledge

To expand the knowledge base:

1. Add `.md`, `.txt`, or `.json` files to the `knowledge_base/` folder
2. Restart the server with index rebuild:
```bash
REBUILD_INDEX=true uvicorn main:app --reload --port 8000
```

Or modify `main.py` lifespan to force rebuild: `rag.build_index(force_rebuild=True)`

---

## Project Structure

```
jungle-rag/
├── main.py                 # FastAPI app with endpoints
├── rag.py                  # RAG logic (indexing & retrieval)
├── requirements.txt        # Python dependencies
├── railway.toml           # Railway deployment config
├── knowledge_base/        # Survival guide documents
├── static/                # Frontend HTML/CSS/JS
└── .chromadb/            # Vector database (gitignored)
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Yes | Free API key from [console.groq.com](https://console.groq.com) |
| `REBUILD_INDEX` | No | Set to `true` to force rebuild vector index on startup |

---

## License

MIT

---

## Contributing

Contributions are welcome! Feel free to:
- Add new survival guides to `knowledge_base/`
- Improve the UI
- Enhance the RAG pipeline
- Fix bugs

---

## Acknowledgments

- Groq for free Llama 3.1 inference
- ChromaDB for vector storage
- sentence-transformers for embeddings
