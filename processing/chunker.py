#LangChain's splitter that tries to break text at natural boundaries (paragraphs, sentences, words) before falling back to hard character cuts



from langchain_text_splitters import RecursiveCharacterTextSplitter


class Chunker: 
    def __init__(self, chunk_size=512, overlap=64):  #chunk size is max character har chunk rakhsakta, overlap yaani neighbouring chunks ke paas kitne same characters rehsakte - here 64, meaning chunk1 ke last 64 cars chunk2 ke start rehte
    
        self.splitter =RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap
        )
    
    def chunk(self, raw_chunks: list) -> list:
        final = []
        for item in raw_chunks:                  #raw chunks is the list of dictionaries from the extractor.
            splits = self.splitter.split_text(item["text"])  #split_text takes a string and returns a list of smaller strings according to the size and overlap
            for j, split in enumerate(splits):
                final.append({
                    "text": split, 
                    "metadata":{
                        "source":item["source"],
                        "modality":item["modality"],
                        **{k: v for k, v in item.items() if k != "text"},
                        "chunk_index":j
                    }
                })

        return final
    


#basic flow is 

# extractor gives dictionaries -> chunker loops over each item -> pulls out ["text"] -> split_text() gets it splits into smaller strings