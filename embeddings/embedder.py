from langchain_huggingface import HuggingFaceEmbeddings

class Embedder: 
    def __init__(self):
        self.model= HuggingFaceEmbeddings(
            model_name= "sentence-transformers/all-MiniLM-L6-v2"
        )

    def embed(self, texts: list) -> list: 
        return self.model.embed_documents(texts)  #each vector is a list of 384 floats for this model 
    
    def embed_query(self, query: str) -> list:
        return self.model.embed_query(query)