import argparse
from typing import List, Tuple

from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain.schema.document import Document

from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma_db"

PROMPT_TEMPLATE = """
Answer the question based only on the following context from the Harry Potter book series:

{context}

---

Question: {question}
Answer as if you are an AI assistant well-versed in the Harry Potter universe:
"""

def setup_chroma() -> Chroma:
    """
    Set up and return a Chroma vector store instance.

    Returns:
        Chroma: An instance of the Chroma vector store.
    """
    embedding_function = get_embedding_function()
    return Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

def search_documents(db: Chroma, query: str, k: int = 5) -> List[Tuple[Document, float]]:
    """
    Search for relevant documents in the vector store.

    Args:
        db (Chroma): The Chroma vector store instance.
        query (str): The query text to search for.
        k (int): The number of results to return.

    Returns:
        List[Tuple[Document, float]]: A list of tuples containing the relevant documents and their scores.
    """
    return db.similarity_search_with_score(query, k=k)

def generate_response(context: str, query: str, model_name: str) -> str:
    """
    Generate a response using the LLM based on the context and query.

    Args:
        context (str): The context information for the query.
        query (str): The user's query.
        model_name (str): The name of the Ollama model to use.

    Returns:
        str: The generated response.
    """
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context, question=query)
    
    model = Ollama(model=model_name)
    return model.invoke(prompt, temperature=0.7)

def query_rag(query_text: str, model_name: str) -> str:
    """
    Perform a RAG query and return the response.

    Args:
        query_text (str): The query text from the user.
        model_name (str): The name of the Ollama model to use.

    Returns:
        str: The formatted response including sources.
    """
    db = setup_chroma()
    results = search_documents(db, query_text)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    response_text = generate_response(context_text, query_text, model_name)

    sources = [doc.metadata.get("source", "Unknown") + f" (Page {doc.metadata.get('page', 'Unknown')})" for doc, _score in results]
    formatted_response = f"{response_text}\nSources: {', '.join(sources)}"
    return formatted_response

def main():
    """
    Main function to handle command-line querying.
    """
    parser = argparse.ArgumentParser(description="Query the Harry Potter RAG system.")
    parser.add_argument("query_text", type=str, help="The query text.")
    parser.add_argument("--model", type=str, default="llama2", help="The Ollama model to use.")
    args = parser.parse_args()
    print(query_rag(args.query_text, args.model))

if __name__ == "__main__":
    main()