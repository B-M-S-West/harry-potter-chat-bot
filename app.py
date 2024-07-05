import streamlit as st
import os
import ollama
from populate_database import load_documents, split_documents, add_to_chroma, CHROMA_PATH
from query_data import query_rag

def extract_model_names(models_info):
    """Extract model names from the provided models information."""
    return tuple(model["name"] for model in models_info["models"])

def initialize_database():
    """Initialize the database if it doesn't exist."""
    if not os.path.exists(CHROMA_PATH):
        st.info("Initializing database. This may take a moment...")
        documents = load_documents()
        chunks = split_documents(documents)
        add_to_chroma(chunks)
        st.success("Database initialized successfully!")
    else:
        st.success("Database already exists.")

def get_rag_response(question: str, selected_model: str) -> str:
    """Get a response from the RAG system."""
    return query_rag(question, selected_model)

def display_chat_history(message_container):
    """Display the chat history."""
    for message in st.session_state["messages"]:
        avatar = "🪄" if message["role"] == "assistant" else "🧙‍♂️"
        with message_container.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

def main():
    st.set_page_config(
        page_title="Harry Potter RAG Chatbot",
        page_icon="🪄",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.subheader("⚡🧙‍♂️🪄 Harry Potter RAG Chatbot", divider="gray", anchor=False)

    try:
        models_info = ollama.list()
        available_models = extract_model_names(models_info)
    except Exception as e:
        st.error("Unable to fetch available models. Please check your Ollama installation.")
        return

    col1, col2 = st.columns([1.5, 2])

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    selected_model = col1.selectbox(
        "Select a model for questioning:",
        available_models,
        index=available_models.index("llama2") if "llama2" in available_models else 0
    )

    # Check for different image formats
    image_formats = ['jpg', 'png', 'jpeg', 'gif']
    image_path = next((f"static/harry_potter.{ext}" for ext in image_formats if os.path.exists(f"static/harry_potter.{ext}")), None)

    if image_path:
        col1.image(image_path, width=300)
    else:
        col1.warning("Harry Potter image not found. Place an image named 'harry_potter' with a common image extension in the 'static' folder.")

    # Initialize database
    initialize_database()

    delete_collection = col1.button("⚠️ Clear Chat History", type="secondary")

    if delete_collection:
        st.session_state["messages"] = []
        st.rerun()

    with col2:
        message_container = st.container(height=500, border=True)
        display_chat_history(message_container)

        if prompt := st.chat_input("Ask a question about Harry Potter..."):
            st.session_state["messages"].append({"role": "user", "content": prompt})
            message_container.chat_message("user", avatar="🧙‍♂️").markdown(prompt)

            with message_container.chat_message("assistant", avatar="🪄"):
                with st.spinner("🪄 Casting a spell to find the answer..."):
                    response = get_rag_response(prompt, selected_model)
                    
                    # Split the response into the actual response and sources
                    response_text, sources = response.split("Sources:", 1)
                    response_text = response_text.strip()
                    
                    st.markdown(response_text)
                    
                    with st.expander("View Sources"):
                        st.write(sources.strip())

            st.session_state["messages"].append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()