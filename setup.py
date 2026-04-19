from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="mmrag",
    version="0.1.0",
    author="Juveria",
    description="A modular multi-modal RAG framework for ingesting and querying PDFs, images, audio, and video.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jaarso0/MultiModal-RAG",
    packages=find_packages(exclude=["venv", "faiss_index", "uploads"]),
    python_requires=">=3.10",
    install_requires=[
        "fastapi",
        "uvicorn",
        "python-multipart",
        "langchain",
        "langchain-community",
        "langchain-huggingface",
        "langchain-google-genai",
        "langchain-core",
        "faiss-cpu",
        "openai-whisper",
        "pytesseract",
        "pdf2image",
        "PyPDF2",
        "opencv-python",
        "sentence-transformers",
        "python-dotenv",
        "google-genai",
        "Pillow",
        "transformers==4.38.2",
        "timm",
        "einops",
        "torch",
        "boto3",
        "pydantic-settings",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)