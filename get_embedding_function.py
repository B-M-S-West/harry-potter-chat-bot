# from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding_function():
    """
    Create and return a HuggingFaceEmbeddings instance for faster text embedding.

    Returns:
        Embeddings: An instance of HuggingFaceEmbeddings using a fast, lightweight model.
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return embeddings

# def get_embedding_function():
#     """
#     Create and return an OllamaEmbeddings instance for text embedding.

#     Returns:
#         Embeddings: An instance of OllamaEmbeddings using the 'nomic-embed-text' model.
#     """
#     embeddings = OllamaEmbeddings(model="nomic-embed-text", show_progress=True)
#     return embeddings