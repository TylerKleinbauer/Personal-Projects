################################################################################
# Argusa AI Assistant - Streamlit Application
# 
# This application creates a chatbot that answers questions about Argusa using RAG
# (Retrieval-Augmented Generation). It loads content from Argusa's website,
# processes it, and uses GPT-4 to generate responses based on the retrieved content.
################################################################################

# Standard library imports
import os

# Third-party imports
import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

################################################################################
# Streamlit Configuration
################################################################################

st.set_page_config(page_title="Argusa AI Assistant")
st.title("Argusa AI Assistant")

################################################################################
# Document Processing Functions
################################################################################

def load_web_pages(urls):
    """Load and process web pages from the provided URLs.
    
    Args:
        urls (list): List of URLs to load content from
        
    Returns:
        list: List of Document objects containing the loaded content
    """
    loader = WebBaseLoader(urls)
    return loader.load()

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """Split documents into smaller chunks for processing.
    
    Args:
        documents (list): List of Document objects to split
        chunk_size (int): Maximum size of each chunk
        chunk_overlap (int): Number of characters to overlap between chunks
        
    Returns:
        list: List of split Document objects
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(documents)

def create_vector_store(texts, embeddings):
    """Create a FAISS vector store from the processed documents.
    
    Args:
        texts (list): List of text chunks to store
        embeddings: Embedding model to use for vectorization
        
    Returns:
        FAISS: Vector store containing the document embeddings
    """
    return FAISS.from_documents(texts, embeddings)

################################################################################
# Retrieval System
################################################################################

def create_ensemble_retriever(docs, embeddings):
    """Create an ensemble retriever combining BM25 and vector search.
    
    This function creates a hybrid retrieval system that combines:
    1. BM25: Traditional keyword-based search
    2. Vector Search: Semantic similarity search
    
    Args:
        docs (list): List of documents to index
        embeddings: Embedding model for vector search
        
    Returns:
        EnsembleRetriever: Combined retrieval system
    """
    # Split documents into chunks
    texts = split_documents(docs)
    
    # Create vector store and retriever
    vector_store = create_vector_store(texts, embeddings)
    vector_retriever = vector_store.as_retriever()
    
    # Create BM25 retriever
    bm25_retriever = BM25Retriever.from_texts([t.page_content for t in texts])
    
    # Combine retrievers with equal weights
    return EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]
    )

################################################################################
# Conversation Chain
################################################################################

def create_chain(retriever, openai_api_key, chat_memory):
    """Create the full conversation chain for the chatbot.
    
    This function sets up:
    1. The language model (GPT-4)
    2. The conversation prompt template
    3. The memory system for maintaining context
    4. The complete processing chain
    
    Args:
        retriever: Document retriever for RAG
        openai_api_key (str): OpenAI API key
        chat_memory: Streamlit chat message history
        
    Returns:
        Chain: Complete conversation processing chain
    """
    # Initialize the language model
    model = ChatOpenAI(
        temperature=0.7,  # Controls response creativity (0.0 = focused, 1.0 = creative)
        model="gpt-4o-mini",
        openai_api_key=openai_api_key
    )
    
    # Create the prompt template for the assistant
    system_prompt = """You are a helpful AI assistant for busy professionals. You help users learn more about Argusa, a data consulting company.
    Use the following context and the users' chat history to help the user.
    If you don't know the answer, just say that you don't know. 
    Always be professional and courteous.
    
    Context: {context}
    
    Question: {question}"""
    
    prompt = ChatPromptTemplate.from_template(system_prompt)
    
    # Set up conversation memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        chat_memory=chat_memory,
        return_messages=True
    )
    
    # Create and return the complete chain
    def get_context(input_dict):
        query = input_dict["question"]
        docs = retriever.get_relevant_documents(query)
        return "\n\n".join(doc.page_content for doc in docs)
    
    chain = (
        {"context": get_context, "question": lambda x: x["question"]}
        | prompt
        | model
    )
    
    return chain

################################################################################
# Streamlit Interface
################################################################################

def show_chat_interface(chain):
    """Display and handle the chat interface in Streamlit.
    
    This function:
    1. Maintains chat message history
    2. Displays the chat interface
    3. Handles user input
    4. Generates and displays responses
    
    Args:
        chain: The conversation processing chain
    """
    # Initialize chat history if not exists
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How may I help you learn about Argusa?"}]

    # Display existing chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Handle new user input
    if prompt := st.chat_input():
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Generate and display assistant response
        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = chain.invoke({"question": prompt})
                    st.markdown(response.content)
            message = {"role": "assistant", "content": response.content}
            st.session_state.messages.append(message)

@st.cache_resource
def initialize_retriever(openai_api_key):
    """Initialize the document retriever with Argusa website content.
    
    This function:
    1. Defines the URLs to scrape
    2. Loads the web content
    3. Creates the embedding model
    4. Initializes the retriever
    
    Args:
        openai_api_key (str): OpenAI API key for embeddings
        
    Returns:
        EnsembleRetriever: Initialized retriever with loaded content
    """
    # Define Argusa website URLs to load
    urls = [
        "https://www.argusa.ch/",
        "https://www.argusa.ch/service-lines",
        "https://www.argusa.ch/service-lines/entreprise-analytics",
        "https://www.argusa.ch/service-lines/governance-strategy",
        "https://www.argusa.ch/service-lines/data-literacy",
        "https://www.argusa.ch/industries",
        "https://www.argusa.ch/insights",
        "https://www.argusa.ch/events",
        "https://www.argusa.ch/about-us",
        "https://www.argusa.ch/contact-us",
        "https://www.argusa.ch/career"
    ]
    
    # Load documents and create embeddings
    docs = load_web_pages(urls)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model="text-embedding-3-small")
    
    return create_ensemble_retriever(docs, embeddings)

################################################################################
# Main Application
################################################################################

def main():
    """Main application logic.
    
    This function:
    1. Handles API key setup
    2. Initializes the retriever and conversation chain
    3. Displays the chat interface
    """
    # Handle OpenAI API key
    with st.sidebar:
        if "OPENAI_API_KEY" in st.secrets:
            openai_api_key = st.secrets["OPENAI_API_KEY"]
            st.success("OpenAI API key found in secrets!")
        else:
            openai_api_key = st.text_input(
                "OpenAI API Key",
                type="password",
                help="Get your API key from https://platform.openai.com/account/api-keys"
            )
            if not openai_api_key:
                st.warning("Please enter your OpenAI API key to continue.")
                st.stop()
            st.session_state["OPENAI_API_KEY"] = openai_api_key

    # Initialize components
    retriever = initialize_retriever(openai_api_key)
    chain = create_chain(
        retriever,
        openai_api_key,
        StreamlitChatMessageHistory(key="langchain_messages")
    )
    
    # Display chat interface
    show_chat_interface(chain)

# Application entry point
if __name__ == "__main__":
    main()