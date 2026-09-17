from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List

from langchain_core.documents import Document
from dotenv import load_dotenv


load_dotenv()

def chunk_document(document: List[Document], chunk_size=200, chunk_overlap=20) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(document)
    print(f"Number of chunks created: {len(chunks)}")
    return chunks

