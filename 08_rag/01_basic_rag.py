"""
Basic RAG (Retrieval Augmented Generation) Implementation
Simple RAG system using LangChain and Chroma.
"""

# Note: Install required packages:
# pip install langchain chromadb openai sentence-transformers

from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
import os


def create_basic_rag_system(use_openai=True):
    """
    Create a basic RAG system
    
    Args:
        use_openai: If True, use OpenAI embeddings (requires API key)
                   If False, use free HuggingFace embeddings
    """
    print("=" * 60)
    print("Building Basic RAG System")
    print("=" * 60)
    
    # Sample documents (in practice, load from files)
    documents = [
        "Machine Learning is a subset of artificial intelligence that enables systems to learn from data.",
        "Deep Learning uses neural networks with multiple layers to learn complex patterns.",
        "Natural Language Processing helps computers understand and process human language.",
        "Large Language Models like GPT are trained on vast amounts of text data.",
        "RAG combines retrieval of relevant documents with language model generation."
    ]
    
    print("\n1. Splitting documents into chunks...")
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20
    )
    
    # For this example, we'll create chunks from our documents
    chunks = []
    for doc in documents:
        chunks.extend(text_splitter.split_text(doc))
    
    print(f"   Created {len(chunks)} chunks")
    
    print("\n2. Creating embeddings...")
    # Create embeddings
    if use_openai:
        # Requires OPENAI_API_KEY environment variable
        if not os.getenv("OPENAI_API_KEY"):
            print("   Warning: OPENAI_API_KEY not set, using HuggingFace embeddings")
            embeddings = HuggingFaceEmbeddings()
        else:
            embeddings = OpenAIEmbeddings()
    else:
        # Free alternative using HuggingFace
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    
    print("   Embeddings created")
    
    print("\n3. Creating vector store...")
    # Create vector store
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"  # Optional: persist to disk
    )
    
    print("   Vector store created")
    
    print("\n4. Setting up retrieval chain...")
    # Create retrieval QA chain
    if use_openai and os.getenv("OPENAI_API_KEY"):
        llm = OpenAI(temperature=0)
    else:
        # For demo, we'll use a simple approach
        # In practice, you'd use a local LLM or API
        print("   Note: Using simple retrieval (LLM requires API key)")
        llm = None
    
    if llm:
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
            return_source_documents=True
        )
        
        return qa_chain, vectorstore
    else:
        # Return just the retriever for demonstration
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
        return retriever, vectorstore


def query_rag_system(qa_chain, query):
    """Query the RAG system"""
    if hasattr(qa_chain, 'run'):
        # Full QA chain
        result = qa_chain.run(query)
        return result
    else:
        # Just retrieval (for demo)
        docs = qa_chain.get_relevant_documents(query)
        return docs


def demonstrate_rag():
    """Demonstrate RAG system"""
    print("\n" + "=" * 60)
    print("RAG System Demonstration")
    print("=" * 60)
    
    # Create RAG system (using free embeddings)
    retriever, vectorstore = create_basic_rag_system(use_openai=False)
    
    # Example queries
    queries = [
        "What is machine learning?",
        "How does deep learning work?",
        "What is RAG?",
    ]
    
    print("\n" + "-" * 60)
    print("Querying RAG System")
    print("-" * 60)
    
    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 60)
        
        # Retrieve relevant documents
        docs = retriever.get_relevant_documents(query)
        
        print(f"\nRetrieved {len(docs)} relevant chunks:")
        for i, doc in enumerate(docs, 1):
            print(f"\nChunk {i}:")
            print(f"  {doc.page_content[:150]}...")
        
        # In a full RAG system, these chunks would be passed to LLM
        # along with the query to generate an answer
        print("\n[In full RAG: These chunks + query → LLM → Answer]")


def rag_without_llm_demo():
    """Demonstrate RAG retrieval without LLM (for learning)"""
    print("\n" + "=" * 60)
    print("RAG Components Explained")
    print("=" * 60)
    
    print("""
    RAG Pipeline:
    
    1. DOCUMENT LOADING
       - Load documents (PDF, TXT, etc.)
    
    2. CHUNKING
       - Split documents into smaller chunks
       - Why? LLMs have context limits
    
    3. EMBEDDING
       - Convert chunks to vectors
       - Enables semantic search
    
    4. VECTOR STORE
       - Store embeddings in database
       - Enables fast similarity search
    
    5. RETRIEVAL
       - Find relevant chunks for query
       - Based on semantic similarity
    
    6. GENERATION
       - Pass query + retrieved chunks to LLM
       - LLM generates answer using context
    """)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Basic RAG System")
    print("=" * 60)
    
    # Show RAG components
    rag_without_llm_demo()
    
    # Demonstrate basic RAG
    try:
        demonstrate_rag()
        
        print("\n" + "=" * 60)
        print("Next Steps:")
        print("1. Add LLM for full RAG (requires API key)")
        print("2. Try different vector databases (Pinecone, Weaviate)")
        print("3. Experiment with chunking strategies")
        print("4. Add re-ranking for better results")
        print("5. Build production RAG system")
        print("=" * 60)
        
    except ImportError as e:
        print(f"\nMissing package: {e}")
        print("\nInstall required packages:")
        print("pip install langchain chromadb sentence-transformers")
        print("\nFor OpenAI (optional):")
        print("pip install openai")
        print("export OPENAI_API_KEY='your-key'")

