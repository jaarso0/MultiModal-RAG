import os
from dotenv import load_dotenv

load_dotenv()

TESSERACT_PATH = os.getenv("TESSERACT_PATH")
POPPLER_PATH = os.getenv("POPPLER_PATH")
FFMPEG_PATH = os.getenv("FFMPEG_PATH")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 64))
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./faiss_index")

GEMINI_API_KEY_LLM = os.getenv("GEMINI_API_KEY_LLM")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_API_KEY_VISION = os.getenv("GEMINI_API_KEY_VISION")