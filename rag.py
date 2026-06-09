from dotenv import load_dotenv
import os

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize the LLM (using OpenAI)
llm = OpenAI(openai_api_key=openai_api_key)


# rag.py
retriever = None

def initialize_rag():
    global retriever

    loader = TextLoader('data/my_document.txt')
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(
        openai_api_key=openai_api_key
    )

    vector_store = FAISS.from_documents(chunks, embeddings)

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )


# Function to set up the RAG system
def setup_rag_system():
    # Load the document
    loader = TextLoader('data/my_document.txt')
    documents = loader.load()

    # Split the document into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    document_chunks = splitter.split_documents(documents)

    # Initialize embeddings with OpenAI API key
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # Create FAISS vector store from document chunks and embeddings
    vector_store = FAISS.from_documents(document_chunks, embeddings)

    # Return the retriever for document retrieval with specified search_type
    retriever = vector_store.as_retriever(
        search_type="similarity",  # or "mmr" or "similarity_score_threshold"
        search_kwargs={"k": 5}  # Adjust the number of results if needed
    )
    return retriever
    

# Function to get the response from the RAG system
async def get_rag_response(query: str):
    retriever = setup_rag_system()

    # Retrieve the relevant documents using 'get_relevant_documents' method
    retrieved_docs = retriever.invoke(query)

    # Prepare the input for the LLM: Combine the query and the retrieved documents into a single string
    context = "\n".join([doc.page_content for doc in retrieved_docs])

    # LLM expects a list of strings (prompts), so we create one by combining the query with the retrieved context
    prompt = [f"Use the following information to answer the question:\n\n{context}\n\nQuestion: {query}"]

    # Generate the final response using the language model (LLM)
    generated_response = llm.generate(prompt)  # Pass as a list of strings
    
    return generated_response













