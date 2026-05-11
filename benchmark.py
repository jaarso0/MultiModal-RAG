import asyncio
import time
from ingestion.pdf_extractor import PDFExtractor
from ingestion.audio_extractor import AudioExtractor
from ingestion.image_extractor import ImageExtractor
from processing.chunker import Chunker
from embeddings.embedder import Embedder
from vector_store.faiss_store import FAISSStore
from llm.chain import build_chain

# Test cases — question + which file should be the source
TEST_CASES = [
    ("what is the document about?", "test.pdf"),
    ("salt pickle tastes best with?", "test.wav"),
    ("why did the prophet go to ta'if?", "test.png"),
]

async def benchmark_ingestion():
    print("\n=== INGESTION BENCHMARK ===\n")
    results = []
    chunker = Chunker()
    
    files = [
        ("test.pdf", PDFExtractor()),
        ("test.png", ImageExtractor()),
        ("test.wav", AudioExtractor()),
    ]
    
    for file_path, extractor in files:
        start = time.time()
        raw = await extractor.extract(file_path)
        chunks = chunker.chunk(raw)
        elapsed = round(time.time() - start, 2)
        results.append({
            "file": file_path,
            "time": f"{elapsed}s",
            "chunks": len(chunks)
        })
        print(f"  {file_path:<20} {elapsed}s   {len(chunks)} chunks")
    
    return results

def benchmark_retrieval(store):
    print("\n=== RETRIEVAL BENCHMARK ===\n")
    correct = 0
    for question, expected_source in TEST_CASES:
        results = store.search(question, k=5)
        sources = [r.metadata["source"] for r in results]
        hit = any(expected_source in s for s in sources)
        if hit:
            correct += 1
        status = "✅" if hit else "❌"
        print(f"  {status} '{question[:40]}...'")
    
    precision = round(correct / len(TEST_CASES) * 100, 1)
    print(f"\n  Retrieval Precision: {precision}%")
    return precision

def benchmark_query(store):
    print("\n=== QUERY LATENCY BENCHMARK ===\n")
    chain, retriever = build_chain(store)
    
    for question, _ in TEST_CASES:
        start = time.time()
        answer = chain.invoke(question)
        elapsed = round(time.time() - start, 2)
        print(f"  {elapsed}s  →  '{question[:40]}...'")
        print(f"           Answer: {answer[:100]}...\n")

async def main():
    # Fresh ingest
    ingestion_results = await benchmark_ingestion()
    
    # Build fresh store from scratch
    embedder = Embedder()
    store = FAISSStore(embedder)
    chunker = Chunker()
    
    files = [
        ("test.pdf", PDFExtractor()),
        ("test.png", ImageExtractor()),
        ("test.wav", AudioExtractor()),
    ]
    
    for file_path, extractor in files:
        raw = await extractor.extract(file_path)
        chunks = chunker.chunk(raw)
        store.add_chunks(chunks)
    
    store.save()
    
    # Now benchmark retrieval and query on fresh store
    benchmark_retrieval(store)
    benchmark_query(store)

asyncio.run(main())