"""
Advanced RAG Techniques
Building production-ready RAG systems with advanced features.
"""

# Note: Install required packages:
# pip install langchain chromadb sentence-transformers openai

from langchain.text_splitter import RecursiveCharacterTextSplitter, SemanticChunking
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("ADVANCED RAG TECHNIQUES - Theory & Practice")
print("=" * 70)

# ============================================================================
# THEORY: Advanced RAG
# ============================================================================
print("\n" + "=" * 70)
print("THEORY: Advanced RAG Concepts")
print("=" * 70)

theory = """
ADVANCED RAG TECHNIQUES

1. CHALLENGES WITH BASIC RAG:
   - Retrieval quality
   - Context window limits
   - Irrelevant chunks
   - Hallucination
   - No source attribution

2. ADVANCED TECHNIQUES:

   a) IMPROVED CHUNKING
      - Semantic chunking (by meaning)
      - Overlap strategies
      - Metadata preservation
   
   b) RE-RANKING
      - Re-order retrieved chunks
      - Cross-encoder models
      - Better relevance
   
   c) HYBRID RETRIEVAL
      - Dense + Sparse retrieval
      - BM25 + Embeddings
      - Best of both worlds
   
   d) QUERY EXPANSION
      - Expand query with related terms
      - Generate multiple queries
      - Better retrieval
   
   e) MULTI-HOP RETRIEVAL
      - Multiple retrieval steps
      - Iterative refinement
      - Complex queries

3. PRODUCTION CONSIDERATIONS:
   - Error handling
   - Monitoring
   - Caching
   - Scalability
   - Evaluation
"""

print(theory)

# ============================================================================
# PRACTICE: Advanced Chunking Strategies
# ============================================================================
print("\n" + "=" * 70)
print("PRACTICE: Advanced Chunking")
print("=" * 70)

# Sample document
long_document = """
Machine Learning is a subset of artificial intelligence that focuses on algorithms 
that can learn from data. Deep Learning is a subset of machine learning that uses 
neural networks with multiple layers. Natural Language Processing helps computers 
understand and process human language. Large Language Models like GPT are trained 
on vast amounts of text data. RAG combines retrieval of relevant documents with 
language model generation. Vector databases store embeddings for fast similarity search.
Embeddings convert text to numerical vectors. Semantic search finds documents by meaning.
"""

print(f"\nOriginal document length: {len(long_document)} characters")

# Strategy 1: Recursive Character Splitting (Basic)
print("\n1. Recursive Character Splitting:")
splitter_basic = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    length_function=len
)
chunks_basic = splitter_basic.split_text(long_document)
print(f"Number of chunks: {len(chunks_basic)}")
for i, chunk in enumerate(chunks_basic[:3], 1):
    print(f"  Chunk {i}: {chunk[:60]}...")

# Strategy 2: Semantic Chunking (Advanced)
print("\n2. Semantic Chunking (by meaning):")
print("Note: Requires embeddings model")
print("""
# Semantic chunking groups by meaning
splitter_semantic = SemanticChunking.from_tiktoken_encoder(
    HuggingFaceEmbeddings(),
    chunk_size=100
)
chunks_semantic = splitter_semantic.create_documents([long_document])
# Better preserves context and meaning
""")

# Strategy 3: Metadata Preservation
print("\n3. Chunking with Metadata:")
print("""
# Preserve metadata with chunks
from langchain.schema import Document

documents = [Document(
    page_content=chunk,
    metadata={"source": "ml_guide.pdf", "page": 1, "chunk_id": i}
) for i, chunk in enumerate(chunks)]
# Helps with source attribution
""")

# ============================================================================
# PRACTICE: Re-ranking
# ============================================================================
print("\n" + "=" * 70)
print("Re-ranking Retrieved Chunks")
print("=" * 70)

print("""
Re-ranking improves retrieval quality by re-ordering results:

1. Initial Retrieval: Get top-k chunks (e.g., 20)
2. Re-ranking: Score and re-order (e.g., top-5)
3. Final Context: Use re-ranked chunks

Benefits:
- Better relevance
- Removes irrelevant chunks
- Improves answer quality

Example:

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# Base retriever
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

# Re-ranker
compressor = LLMChainExtractor.from_llm(OpenAI())
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

# Now retrieval returns re-ranked results
docs = compression_retriever.get_relevant_documents(query)
""")

# ============================================================================
# PRACTICE: Hybrid Retrieval
# ============================================================================
print("\n" + "=" * 70)
print("Hybrid Retrieval (Dense + Sparse)")
print("=" * 70)

print("""
Hybrid retrieval combines:
- Dense retrieval (embeddings): Semantic similarity
- Sparse retrieval (BM25): Keyword matching

Benefits:
- Better coverage
- Handles both semantic and keyword queries
- More robust

Example using LangChain:

from langchain.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever

# Dense retriever (embeddings)
dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Sparse retriever (BM25)
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 5

# Ensemble retriever
ensemble_retriever = EnsembleRetriever(
    retrievers=[dense_retriever, bm25_retriever],
    weights=[0.5, 0.5]  # Equal weight
)

# Retrieve
docs = ensemble_retriever.get_relevant_documents(query)
""")

# ============================================================================
# PRACTICE: Query Expansion
# ============================================================================
print("\n" + "=" * 70)
print("Query Expansion")
print("=" * 70)

print("""
Query expansion generates multiple queries from original:

Original: "What is machine learning?"
Expanded:
  - "What is machine learning?"
  - "Define machine learning"
  - "Explain machine learning concepts"
  - "Machine learning overview"

Benefits:
- Better retrieval coverage
- Handles query variations
- More comprehensive results

Example:

from langchain.llms import OpenAI

def expand_query(query):
    llm = OpenAI()
    prompt = f\"\"\"
    Generate 3 alternative phrasings of this query:
    {query}
    
    Return as list:
    \"\"\"
    expanded = llm(prompt)
    queries = [query] + parse_expanded(expanded)
    return queries

# Retrieve for each expanded query
all_docs = []
for q in expand_query(original_query):
    docs = retriever.get_relevant_documents(q)
    all_docs.extend(docs)

# Deduplicate and rank
unique_docs = deduplicate(all_docs)
""")

# ============================================================================
# PRACTICE: Multi-hop RAG
# ============================================================================
print("\n" + "=" * 70)
print("Multi-hop RAG (Iterative Retrieval)")
print("=" * 70)

print("""
Multi-hop RAG performs multiple retrieval steps:

Step 1: Initial retrieval
  Query: "What is RAG?"
  → Retrieve: Documents about RAG

Step 2: Refined retrieval
  Query: "RAG" + context from step 1
  → Retrieve: More specific documents

Step 3: Final answer
  Combine all retrieved context
  → Generate answer

Benefits:
- Handles complex queries
- Better context gathering
- More accurate answers

Example:

def multi_hop_rag(query, max_hops=2):
    context = []
    
    for hop in range(max_hops):
        # Retrieve
        docs = retriever.get_relevant_documents(query)
        context.extend(docs)
        
        # Refine query if not last hop
        if hop < max_hops - 1:
            query = refine_query(query, context)
    
    # Generate answer
    answer = llm.generate(query, context)
    return answer
""")

# ============================================================================
# PRODUCTION RAG SYSTEM
# ============================================================================
print("\n" + "=" * 70)
print("Production RAG System Checklist")
print("=" * 70)

production_checklist = """
ESSENTIAL COMPONENTS:

1. DATA PIPELINE:
   ✓ Document loading (PDF, DOCX, TXT, etc.)
   ✓ Chunking strategy
   ✓ Embedding generation
   ✓ Vector database storage
   ✓ Metadata preservation

2. RETRIEVAL:
   ✓ Embedding model selection
   ✓ Similarity search
   ✓ Re-ranking (optional)
   ✓ Hybrid retrieval (optional)
   ✓ Top-k selection

3. GENERATION:
   ✓ LLM selection
   ✓ Prompt engineering
   ✓ Context formatting
   ✓ Answer generation
   ✓ Source attribution

4. EVALUATION:
   ✓ Retrieval metrics
   ✓ Generation metrics
   ✓ End-to-end testing
   ✓ Human evaluation

5. MONITORING:
   ✓ Query logging
   ✓ Performance metrics
   ✓ Error tracking
   ✓ User feedback

6. OPTIMIZATION:
   ✓ Caching frequent queries
   ✓ Batch processing
   ✓ Async operations
   ✓ Load balancing
"""

print(production_checklist)

# ============================================================================
# KEY TAKEAWAYS
# ============================================================================
print("\n" + "=" * 70)
print("KEY TAKEAWAYS")
print("=" * 70)

takeaways = """
1. Basic RAG is just the start
2. Chunking strategy matters a lot
3. Re-ranking improves quality
4. Hybrid retrieval is more robust
5. Query expansion helps coverage
6. Multi-hop handles complex queries

IMPROVEMENTS OVER BASIC RAG:
- Better chunking → Better context
- Re-ranking → More relevant chunks
- Hybrid retrieval → Better coverage
- Query expansion → More comprehensive
- Multi-hop → Complex queries

PRODUCTION CONSIDERATIONS:
- Error handling
- Monitoring and logging
- Caching
- Scalability
- Evaluation framework

NEXT STEPS:
- Implement these techniques
- Evaluate on your data
- Optimize for your use case
- Deploy to production
"""

print(takeaways)

print("\n" + "=" * 70)
print("Next: Build your production RAG system!")
print("=" * 70)

