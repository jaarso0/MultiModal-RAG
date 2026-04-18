from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import shutil
import os
import asyncio

from ingestion.image_extractor import ImageExtractor
from ingestion.pdf_extractor import PDFExtractor
from ingestion.audio_extractor import AudioExtractor
from ingestion.video_extractor import VideoExtractor
from processing.chunker import Chunker
from embeddings.embedder import Embedder
from vector_store.faiss_store import FAISSStore
from llm.chain import build_chain


router = APIRouter()

embedder= Embedder()
store = FAISSStore(embedder)
chunker = Chunker()

try:
    store.load()
    print("Loaded exisiting FAISS store.")
except:
    print("No existing index found, starting fresh.")

EXTRACTOR_REGISTRY = {
    "png": ImageExtractor(),
    "jpg": ImageExtractor(),
    "jpeg": ImageExtractor(),
    "pdf": PDFExtractor(),
    "mp3": AudioExtractor(),
    "wav": AudioExtractor(),
    "mp4": VideoExtractor(),
    "mov": VideoExtractor(),
}

UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class QueryRequest(BaseModel):
    question: str
    k: int = 5


@router.get("/health")
def health():
    return {"status": "ok", "message": "RAG Framework is running"}

@router.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    ext = file.filename.split(".")[-1].lower()


    if ext not in EXTRACTOR_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Supported: {list(EXTRACTOR_REGISTRY.keys())}"

        )
    

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    
    try:
        extractor = EXTRACTOR_REGISTRY[ext]
        raw = await extractor.extract(file_path)
        chunks = chunker.chunk(raw)
        store.add_chunks(chunks)
        store.save()

        return {
            "status": "success",
            "file": file.filename,
            "modality" : chunks[0]["metadata"]["modality"] if chunks else "unknown",
            "chunks_added": len(chunks)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@router.post("/query")
async def query(request: QueryRequest):
    if store.store is None:
        raise HTTPException(
            status_code=400,
            detail="No documents ingested yet. Please upload files first."
        )
    
    try: 
        chain, retriever= build_chain(store)

        loop = asyncio.get_event_loop()
        answer = await loop.run_in_executor(None, chain.invoke, request.question)
        source_docs= retriever.invoke(request.question)

        return {
            "question": request.question,
            "answer": answer,
            "sources":[{

                "content": doc.page_content[:200],
                "metadata":doc.metadata
            }
            for doc in source_docs
            ]

        }
    except Exception as e:
        raise HTTPException(status_code= 500, detail=str(e))
    
@router.get("/sources")
def sources():
    if store.store is None:
        return {"ingested_files": []}
    

    docs= store.store.docstore._dict.values()
    unique_sources = list(set(
        doc.metadata.get("source","unkown")
        for doc in docs
    ))

    return {
        "ingested_files": unique_sources,
        "total_chunks": len(store.store.docstore._dict)
    }
    
    