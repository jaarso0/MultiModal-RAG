from ingestion.image_extractor import ImageExtractor
from ingestion.pdf_extractor import PDFExtractor
from ingestion.audio_extractor import AudioExtractor
from ingestion.video_extractor import VideoExtractor
from processing.chunker import Chunker
from embeddings.embedder import Embedder
from vector_store.faiss_store import FAISSStore
from llm.chain import build_chain
import asyncio

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

DEMO_FILES = [
    "test.pdf",
    "test.png",
    "test.wav",
    "test.mp4",
]
async def ingest_all(store: FAISSStore):
    chunker = Chunker()
    for file_path in DEMO_FILES:
        ext = file_path.split(".")[-1].lower()
        extractor = EXTRACTOR_REGISTRY.get(ext)
        if not extractor:
            print(f"No extractor for {file_path}, skipping.")
            continue
        print(f"\nIngesting {file_path}...")
        raw = await extractor.extract(file_path)
        chunks = chunker.chunk(raw)
        store.add_chunks(chunks)
        print(f"Done — {len(chunks)} chunks added from {file_path}")
    store.save()
    print("\nAll files ingested and saved.")




async def main():
    embedder = Embedder()
    store = FAISSStore(embedder)

    
    await ingest_all(store)

   
    chain, retriever = build_chain(store)

    print("\n--- MULTI-MODAL RAG DEMO ---\n")
    while True:
        question = input("Ask a question (or type 'exit'): ")
        if question.lower() == "exit":
            break

        answer = chain.invoke(question)
        source_docs = retriever.invoke(question)

        print(f"\nAnswer:\n{answer}")
        print("\nSources:")
        for doc in source_docs:
            print(f"  - {doc.metadata}")
        print()



asyncio.run(main())