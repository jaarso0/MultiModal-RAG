from langchain_community.vectorstores import FAISS   #faiss handles the embedding-to-index plumbing
from langchain_core.documents import Document  #document is langchain standard data container


class FAISSStore:
    def __init__ (self, embedder):
        self.embedder = embedder.model
        self.store = None

    def add_chunks(self, chunks: list):
        docs = [
            Document(page_content = c["text"], metadata=c["metadata"])  #Document is a prebuilt LangChain container, FAISS is lc components and lc components only talk to each other using Document object 
            for c in chunks
        ]

        if self.store is None:
            self.store = FAISS.from_documents(docs, self.embedder)   #from_documents creates brand new faiss index from scratch 
        else: 
            self.store.add_documents(docs)

    def search(self, query: str, k: int =5):
        return self.store.similarity_search(query, k=k)

    def save(self, path: str = "./faiss_index"):
        self.store.save_local(path)

    def load(self, path: str = "./faiss_index"):
        self.store = FAISS.load_local(
            path, 
            self.embedder,
            allow_dangerous_deserialzation = True
        )

#FAISS is a vector storage and seo that cant read text on its own 
#so it needs an embedding model handed to it. apan huggingface model Embedder class me wrap kardiye and passing to faiss
#faiss uses the hugging face model to internallt convert both docs and queries into vectors whenever needed
