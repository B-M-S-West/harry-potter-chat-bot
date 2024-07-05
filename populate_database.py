import argparse
import os
import shutil
from typing import List

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_community.vectorstores import Chroma

from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma_db"
DATA_PATH = "data"

def load_documents() -> List[Document]:
    """
    Load PDF documents from the DATA_PATH directory.

    Returns:
        List[Document]: A list of loaded documents.
    """
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

def split_documents(documents: List[Document]) -> List[Document]:
    """
    Split the documents into smaller chunks.

    Args:
        documents (List[Document]): The list of documents to split.

    Returns:
        List[Document]: A list of document chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def calculate_chunk_ids(chunks: List[Document]) -> List[Document]:
    """
    Calculate and add unique IDs to document chunks.

    Args:
        chunks (List[Document]): The list of document chunks.

    Returns:
        List[Document]: The list of document chunks with added IDs.
    """
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0
        
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        chunk.metadata["id"] = chunk_id
    return chunks

def add_to_chroma(chunks: List[Document]):
    """
    Add new document chunks to the Chroma vector store.

    Args:
        chunks (List[Document]): The list of document chunks to add.
    """
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function()
    )
    chunks_with_ids = calculate_chunk_ids(chunks)

    existing_items = db.get()
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]

    if new_chunks:
        print(f"Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("No new documents to add.")

def clear_database():
    """
    Clear the existing Chroma database.
    """
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print("Database cleared.")

def main():
    """
    Main function to handle document processing and database population.
    """
    parser = argparse.ArgumentParser(description="Populate or reset the Chroma database with Harry Potter PDF documents.")
    parser.add_argument("--reset", action="store_true", help="Reset the database before populating")
    args = parser.parse_args()

    if args.reset:
        print("Resetting the database...")
        clear_database()

    print("Loading documents...")
    documents = load_documents()
    print("Splitting documents...")
    chunks = split_documents(documents)
    print("Adding documents to Chroma...")
    add_to_chroma(chunks)
    print("Database population complete.")

if __name__ == "__main__":
    main()