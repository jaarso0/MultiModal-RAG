from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from config import GEMINI_API_KEY_LLM, GEMINI_MODEL

def build_chain(vector_store):
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GEMINI_API_KEY_LLM,
        temperature=0
    )

    retriever = vector_store.store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    prompt = PromptTemplate.from_template("""
    You are an intelligent assistant. Use the following retrieved context to answer the question.
    Always mention which source file and modality the answer came from.
    If you don't know the answer from the context, say so honestly.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """)

    def format_docs(docs):
        return "\n\n".join([
            f"[Source: {d.metadata.get('source', 'unknown')} | Modality: {d.metadata.get('modality', 'unknown')}]\n{d.page_content}"
            for d in docs
        ])

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever