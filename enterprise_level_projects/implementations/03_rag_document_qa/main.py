"""
Enterprise Document Q&A System
RAG-based system for answering questions from documents.
"""

# Note: Install required packages:
# pip install langchain chromadb sentence-transformers openai

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("ENTERPRISE DOCUMENT Q&A SYSTEM (RAG)")
print("=" * 70)

# ============================================================================
# DOCUMENT PREPARATION
# ============================================================================
print("\n" + "=" * 70)
print("Step 1: Document Preparation")
print("=" * 70)

# Sample documents (in production, load from files)
documents = [
    """
    Machine Learning is a subset of artificial intelligence that focuses on algorithms 
    that can learn from data. It enables systems to automatically learn and improve 
    from experience without being explicitly programmed.
    """,
    """
    Deep Learning uses neural networks with multiple layers to learn complex patterns 
    in data. It has revolutionized fields like computer vision, natural language processing, 
    and speech recognition.
    """,
    """
    Natural Language Processing (NLP) helps computers understand, interpret, and generate 
    human language. Applications include chatbots, translation, and sentiment analysis.
    """,
    """
    Large Language Models like GPT are trained on vast amounts of text data. They can 
    generate human-like text, answer questions, and perform various language tasks.
    """,
    """
    RAG (Retrieval Augmented Generation) combines retrieval of relevant documents with 
    language model generation. It allows LLMs to access up-to-date information and 
    provide accurate answers with source citations.
    """,
    """
    Vector databases store embeddings for fast similarity search. They enable efficient 
    retrieval of relevant documents based on semantic similarity rather than keyword matching.
    """,
    """
    Embeddings convert text to numerical vectors that capture semantic meaning. Similar 
    texts have similar embeddings, enabling semantic search and retrieval.
    """,
    """
    Semantic search finds documents by meaning rather than exact keyword matches. It 
    uses embeddings to understand query intent and find relevant content.
    """
]

print(f"Prepared {len(documents)} documents")
print(f"Sample document (first 100 chars): {documents[0][:100]}...")

# ============================================================================
# TEXT CHUNKING
# ============================================================================
print("\n" + "=" * 70)
print("Step 2: Text Chunking")
print("=" * 70)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,      # Characters per chunk
    chunk_overlap=50,    # Overlap between chunks
    length_function=len
)

chunks = []
for doc in documents:
    chunks.extend(text_splitter.split_text(doc))

print(f"Created {len(chunks)} chunks from {len(documents)} documents")
print(f"\nSample chunk:")
print(chunks[0][:150] + "...")

# ============================================================================
# EMBEDDING GENERATION
# ============================================================================
print("\n" + "=" * 70)
print("Step 3: Embedding Generation")
print("=" * 70)

# Use free HuggingFace embeddings (no API key needed)
# For production, consider OpenAI embeddings for better quality
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded: sentence-transformers/all-MiniLM-L6-v2")
print("Note: First run will download the model (may take time)")

# ============================================================================
# VECTOR DATABASE CREATION
# ============================================================================
print("\n" + "=" * 70)
print("Step 4: Vector Database Creation")
print("=" * 70)

# Create vector store
vectorstore = Chroma.from_texts(
    texts=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print(f"Vector database created with {len(chunks)} chunks")
print("Saved to: ./chroma_db")

# ============================================================================
# RETRIEVAL SETUP
# ============================================================================
print("\n" + "=" * 70)
print("Step 5: Retrieval Setup")
print("=" * 70)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}  # Retrieve top 3 most relevant chunks
)

print("Retriever configured to return top 3 chunks")

# Test retrieval
test_query = "What is machine learning?"
retrieved_docs = retriever.get_relevant_documents(test_query)

print(f"\nTest Query: '{test_query}'")
print(f"Retrieved {len(retrieved_docs)} relevant chunks:")
for i, doc in enumerate(retrieved_docs, 1):
    print(f"\nChunk {i}:")
    print(f"  {doc.page_content[:150]}...")

# ============================================================================
# RAG CHAIN SETUP
# ============================================================================
print("\n" + "=" * 70)
print("Step 6: RAG Chain Setup")
print("=" * 70)

# Check if OpenAI API key is available
use_openai = os.getenv("OPENAI_API_KEY") is not None

if use_openai:
    print("Using OpenAI for generation (API key found)")
    llm = OpenAI(temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
else:
    print("OpenAI API key not found. Using retrieval-only mode.")
    print("For full RAG, set OPENAI_API_KEY environment variable")
    qa_chain = None

# ============================================================================
# QUESTION ANSWERING
# ============================================================================
print("\n" + "=" * 70)
print("Step 7: Question Answering")
print("=" * 70)

questions = [
    "What is machine learning?",
    "How does RAG work?",
    "What are embeddings?",
    "What is deep learning?"
]

if qa_chain:
    print("\nAnswering questions with RAG...")
    for question in questions:
        print(f"\n{'='*70}")
        print(f"Question: {question}")
        print('='*70)
        
        result = qa_chain({"query": question})
        
        print(f"\nAnswer: {result['result']}")
        print(f"\nSources ({len(result['source_documents'])}):")
        for i, doc in enumerate(result['source_documents'], 1):
            print(f"\n  Source {i}:")
            print(f"    {doc.page_content[:200]}...")
else:
    print("\nRetrieval-only mode (no LLM generation):")
    for question in questions:
        print(f"\n{'='*70}")
        print(f"Question: {question}")
        print('='*70)
        
        docs = retriever.get_relevant_documents(question)
        print(f"\nRetrieved {len(docs)} relevant chunks:")
        for i, doc in enumerate(docs, 1):
            print(f"\n  Chunk {i}:")
            print(f"    {doc.page_content[:200]}...")
        
        print("\n[In full RAG: These chunks would be passed to LLM to generate answer]")

# ============================================================================
# IMPROVEMENTS: RE-RANKING (Conceptual)
# ============================================================================
print("\n" + "=" * 70)
print("Step 8: Advanced Features (Conceptual)")
print("=" * 70)

print("""
Advanced RAG Features to Implement:

1. RE-RANKING:
   - Use cross-encoder models to re-rank retrieved chunks
   - Improves relevance of top results
   - Example: Use sentence-transformers cross-encoder

2. HYBRID RETRIEVAL:
   - Combine dense (embedding) and sparse (BM25) retrieval
   - Better coverage of different query types
   - Example: Use both vector search and keyword search

3. QUERY EXPANSION:
   - Generate multiple query variations
   - Retrieve for each variation
   - Combine results
   - Example: Use LLM to expand queries

4. MULTI-HOP RETRIEVAL:
   - Multiple retrieval steps
   - Use initial results to refine query
   - Better for complex questions
   - Example: Iterative retrieval

5. SOURCE ATTRIBUTION:
   - Track which documents contributed to answer
   - Provide citations
   - Enable fact-checking
   - Example: Include document metadata
""")

# ============================================================================
# API FUNCTION (For Deployment)
# ============================================================================
print("\n" + "=" * 70)
print("Step 9: API Function (For Deployment)")
print("=" * 70)

def answer_question(question, qa_chain=None, retriever=None):
    """Answer a question using RAG"""
    if qa_chain:
        # Full RAG with LLM
        result = qa_chain({"query": question})
        return {
            "answer": result['result'],
            "sources": [doc.page_content for doc in result['source_documents']]
        }
    elif retriever:
        # Retrieval only
        docs = retriever.get_relevant_documents(question)
        return {
            "answer": "Retrieval-only mode. Retrieved relevant chunks.",
            "sources": [doc.page_content for doc in docs]
        }
    else:
        return {"error": "RAG system not initialized"}

print("""
Example API usage:

from fastapi import FastAPI
app = FastAPI()

@app.post("/ask")
def ask_question(question: str):
    result = answer_question(question, qa_chain, retriever)
    return result
""")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"""
✅ RAG System Built!
✅ Documents processed: {len(documents)}
✅ Chunks created: {len(chunks)}
✅ Vector database: ./chroma_db
{'✅ LLM integration: Ready' if qa_chain else '⚠️  LLM integration: Set OPENAI_API_KEY for full RAG'}

Next Steps:
1. Add more documents (PDF, DOCX, etc.)
2. Implement re-ranking for better results
3. Add hybrid retrieval (dense + sparse)
4. Deploy as API (FastAPI)
5. Create web interface
6. Add evaluation metrics
7. Implement monitoring
8. Scale to production
""")

print("=" * 70)

