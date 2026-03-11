import asyncio  #for extract, since its an async function
from ingestion.image_extractor import ImageExtractor
from processing.chunker import Chunker
from embeddings.embedder import Embedder
from vector_store.faiss_store import FAISSStore
from ingestion.pdf_extractor import PDFExtractor
from ingestion.audio_extractor import AudioExtractor
from ingestion.video_extractor import VideoExtractor

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

async def main():
    file_path = "test.mp4"
    ext = file_path.split(".")[-1].lower()

    extractor = EXTRACTOR_REGISTRY.get(ext)
    raw = await extractor.extract(file_path)

    chunker=Chunker()
    chunks = chunker.chunk(raw)

    embedder = Embedder()
    store = FAISSStore(embedder)
    store.add_chunks(chunks)
    store.save()

    results = store.search("how to cut apples")
    for r in results:
        print(r.page_content)
        print(r.metadata)

asyncio.run(main())

