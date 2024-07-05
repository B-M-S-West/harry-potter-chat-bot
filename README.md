# Harry Potter RAG Chatbot 🧙‍♂️🪄

## Overview

This project implements a Harry Potter-themed Retrieval-Augmented Generation (RAG) system for question answering based on the Harry Potter book series. It uses LangChain, Chroma vector store, and Streamlit to create an interactive interface for users to query information from the magical world of Harry Potter.

## Features

- PDF document ingestion and processing of Harry Potter books
- Vector storage using Chroma DB for efficient retrieval of Harry Potter content
- Natural language question answering using various Language Models
- User-friendly web interface with Streamlit, featuring:
  - Interactive chat history with wizard and wand emojis 🧙‍♂️🪄
  - Model selection for different questioning styles
  - Harry Potter-themed layout and design

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/harry-potter-rag-chatbot.git
   cd harry-potter-rag-chatbot
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Prepare your documents:
   - Place your Harry Potter PDF documents in the `data` directory.
   - Ensure you have the rights to use these documents.

5. Install Ollama and download the required models:
   - Follow the instructions at [Ollama's official website](https://ollama.ai/) to install Ollama.
   - Download the models you want to use (e.g., `llama2`) using Ollama's CLI.

## Usage

1. Populate the database (if not already done):
   ```
   python populate_database.py
   ```

2. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

3. Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

4. Use the interface to ask questions about the Harry Potter universe:
   - Select a model from the dropdown menu.
   - Type your question in the chat input.
   - View the AI's response and the sources it used.
   - Enjoy the magical chat history with wizard and wand emojis!

## Project Structure

- `app.py`: Streamlit application for the Harry Potter-themed user interface
- `populate_database.py`: Script to process Harry Potter books and populate the vector store
- `query_data.py`: Functions for querying the RAG system with different models
- `get_embedding_function.py`: Defines the embedding function used for document vectorization
- `static/harry_potter.jpg`: Image file for the user interface

## Customization

- To change the embedding model, modify the `get_embedding_function.py` file.
- To add more Ollama models, download them using Ollama's CLI, and they will automatically appear in the model selection dropdown.
- Adjust the `PROMPT_TEMPLATE` in `query_data.py` to change the AI's personality or knowledge scope.

## Contributing

Contributions to enhance the Harry Potter RAG Chatbot are welcome! Please fork the repository and submit a pull request with your improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project uses [LangChain](https://github.com/hwchase17/langchain) for document processing and LLM integration.
- Vector storage is handled by [Chroma](https://github.com/chroma-core/chroma).
- The web interface is built with [Streamlit](https://streamlit.io/).
- Language models are provided by [Ollama](https://ollama.ai/).
- Special thanks to J.K. Rowling for creating the magical world of Harry Potter that inspired this project.

Note: Ensure you have the necessary rights to use Harry Potter content in your application.