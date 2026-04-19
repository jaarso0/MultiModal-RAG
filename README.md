# Multi-Modal RAG Knowledge Framework

A modular, extensible Retrieval-Augmented Generation (RAG) framework that ingests and semantically indexes content across multiple data modalities - PDFs, scanned documents, images, audio and video using a pluggable extraction pipeline, HuggingFace embeddings, and FAISS vector search.


## What It Does?

Upload any file. Ask any question. Get a grounded answer with sources traced back to the extract file, page, timestamp or frame it came from.


POST /ingest  →  upload PDF, image, audio, or video
POST /query   →  ask a question across everything you've uploaded
GET  /sources →  see all ingested files
GET  /health  →  check server status

## Architecture

RAW File  
    ↓  
Modality-Specific Extractor  (pluggable, extend with BaseExtractor)  
    ↓  
Chunker+Normalizer (consistent, metadata enriched JSON)  
    ↓  
HuggingFace Embedder (swappable embedding model)  
    ↓  
FAISS vector store  
    ↓  
LLM chain  
    ↓  
Answer + Sources  

---

##  Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/jaarso0/MultiModal-RAG
or 

# pip install git+https://github.com/jaarso0/MultiModal-RAG

cd rag-framework
```

### 2. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install system dependencies

| Tool | Purpose | Download |  
|---|---|---|  
| Tesseract | OCR for images and scanned PDFs | [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) |  
| Poppler | PDF to image conversion | [poppler-windows](https://github.com/oschwartz10612/poppler-windows/releases) |  
| FFmpeg | Audio/video processing | [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) |  

### 5. Configure environment

```bash
cp .env.example .env
```

Open `.env` and fill in:
- System tool paths (Tesseract, Poppler, FFmpeg)
- Your LLM API key
- Your preferred models

### 6. Run the API

```bash
python main_api.py
```

Visit `http://localhost:8000/docs` for interactive API documentation.

---

## Using as a Framework

Install directly from GitHub:

```bash
pip install git+https://github.com/jaarso0/MultiModal-RAG
```

Use in your own project:

```python
from mmrag.ingestion.audio_extractor import AudioExtractor
from mmrag.ingestion.video_extractor import VideoExtractor
from mmrag.processing.chunker import Chunker
from mmrag.embeddings.embedder import Embedder
from mmrag.vector_store.faiss_store import FAISSStore
from mmrag.llm.chain import build_chain
import asyncio

async def meeting_summarizer():
    embedder = Embedder()
    store = FAISSStore(embedder)
    chunker = Chunker()

    # Ingest meeting recordings
    audio = AudioExtractor()
    raw = await audio.extract("standup.mp3")
    store.add_chunks(chunker.chunk(raw))

    video = VideoExtractor()
    raw = await video.extract("board_meeting.mp4")
    store.add_chunks(chunker.chunk(raw))

    store.save()

    # Query across all meetings
    chain, _ = build_chain(store)
    answer = chain.invoke("What were the key decisions made?")
    print(answer)

asyncio.run(meeting_summarizer())
```

---

## Swappable Components

Everything is behind an abstraction layer. One config change, nothing breaks.

**Swap the LLM:**
LLM_PROVIDER=gemini    # or openai, claude

**Swap the embedding model:**
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2

**Add a custom extractor:**
```python
from mmrag.ingestion.base import BaseExtractor

class MyExtractor(BaseExtractor):
    async def extract(self, file_path: str):
        # your logic
        return [{"text": "...", "source": file_path, "modality": "custom"}]
```

---


## Project Structure

rag-framework/  
├── ingestion/  
│   ├── base.py                          # Abstract base extractor  
│   ├── image_extractor.py               # OCR for images  
│   ├── pdf_extractor.py                 # Native + scanned PDF  
│   ├── audio_extractor.py               # Whisper STT  
│   └── video_extractor.py               # Florence-2 + Whisper  
├── processing/  
│   └── chunker.py                       # Text splitting + metadata  
├── embeddings/  
│   └── embedder.py                      # HuggingFace embeddings  
├── vector_store/  
│   └── faiss_store.py                   # FAISS wrapper  
├── llm/  
│   └── chain.py                         # LLM-agnostic RAG chain  
├── api/  
│   └── endpoints.py                     # FastAPI routes  
├── storage/  
│   └── object_store.py                  # Local + S3 abstraction  
├── config.py                            # Centralized config  
├── main_api.py                          # FastAPI entry point  
├── demo.py                              # Full pipeline demo  
├── .env.example                         # Configuration template  
└── requirements.txt  


## Contributing

Built as a modular foundation — contributions welcome.

1. Fork the repo
2. Create your feature branch
3. Add your extractor, embedder, or vector store implementation
4. Submit a pull request

---

## LICENSE

MIT License — use it, extend it, build on it.

---

*Built by Juveria Zaheer*
