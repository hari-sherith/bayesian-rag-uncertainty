import os
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentLoader:
    """Loads and chunks text documents for the retrieval pipeline."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def load_documents(self, data_dir: str) -> List[Document]:
        """Scan a directory for .txt files and return LangChain Document objects."""
        data_path = Path(data_dir)
        documents = []
        for txt_file in sorted(data_path.glob("*.txt")):
            text = txt_file.read_text(encoding="utf-8")
            doc = Document(
                page_content=text,
                metadata={"source": str(txt_file)},
            )
            documents.append(doc)
        return documents

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks, preserving and extending metadata."""
        chunks = []
        for doc in documents:
            splits = self.splitter.split_text(doc.page_content)
            for i, split in enumerate(splits):
                chunk = Document(
                    page_content=split,
                    metadata={**doc.metadata, "chunk_id": i},
                )
                chunks.append(chunk)
        return chunks
