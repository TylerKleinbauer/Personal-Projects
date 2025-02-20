# Argusa AI Assistant

A Streamlit-based chatbot that uses RAG (Retrieval-Augmented Generation) to answer questions about Argusa, a data consulting company. The chatbot retrieves information from Argusa's website and uses OpenAI's GPT-4 to generate accurate and contextual responses.

## Features

- Web-based chat interface using Streamlit
- RAG implementation using LangChain
- Ensemble retrieval combining BM25 and vector search
- Conversation memory to maintain context
- Real-time web content retrieval from Argusa's website

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your OpenAI API key:
   - Option 1: Create a `.env` file with:
     ```
     OPENAI_API_KEY=your-api-key
     ```
   - Option 2: Enter the API key in the Streamlit interface when prompted

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```
2. Open your browser at `http://localhost:8501`
3. Start chatting with the AI assistant about Argusa!

## Architecture

- **Document Processing**: Web content is loaded and split into chunks
- **Retrieval**: Uses an ensemble of BM25 and FAISS vector store for efficient search
- **Language Model**: Utilizes OpenAI's GPT-4 for response generation
- **Memory**: Maintains conversation history for contextual responses

## Dependencies

- streamlit
- langchain
- langchain-openai
- langchain-community
- faiss-cpu
- beautifulsoup4
- python-dotenv
- requests
