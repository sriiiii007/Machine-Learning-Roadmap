"""
Enterprise Knowledge Base Assistant
LLM-powered assistant for answering questions from knowledge base.
"""

# Note: Install required packages:
# pip install langchain chromadb sentence-transformers openai

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("ENTERPRISE KNOWLEDGE BASE ASSISTANT")
print("=" * 70)

# ============================================================================
# KNOWLEDGE BASE SETUP
# ============================================================================
print("\n" + "=" * 70)
print("Step 1: Knowledge Base Setup")
print("=" * 70)

# Sample knowledge base content
knowledge_base = {
    "company_policies": """
    Our company values innovation, integrity, and customer satisfaction. 
    Employees are expected to maintain high ethical standards and prioritize 
    customer needs. Remote work is allowed up to 3 days per week.
    """,
    "hr_policies": """
    Vacation days: 20 days per year. Sick leave: 10 days per year. 
    Health insurance covers employees and dependents. 401k matching up to 6%.
    Performance reviews are conducted quarterly.
    """,
    "it_support": """
    IT support is available 24/7. Contact IT helpdesk at it@company.com or 
    call extension 1234. Password reset can be done through the employee portal.
    Software installation requires IT approval.
    """,
    "product_info": """
    Our main product is an AI-powered analytics platform. Key features include 
    real-time dashboards, predictive analytics, and automated reporting. 
    Version 2.0 was released last month with enhanced ML capabilities.
    """,
    "sales_process": """
    Sales process: 1) Lead qualification, 2) Demo presentation, 3) Proposal, 
    4) Negotiation, 5) Contract signing. Average sales cycle is 45 days. 
    Discount approval requires manager approval for >10%.
    """
}

print(f"Knowledge base loaded with {len(knowledge_base)} sections:")
for section in knowledge_base.keys():
    print(f"  - {section}")

# ============================================================================
# DOCUMENT PROCESSING
# ============================================================================
print("\n" + "=" * 70)
print("Step 2: Document Processing")
print("=" * 70)

# Combine all knowledge into documents
documents = []
for section, content in knowledge_base.items():
    documents.append(f"{section.upper()}\n{content}")

# Chunk documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    length_function=len
)

chunks = []
for doc in documents:
    chunks.extend(text_splitter.split_text(doc))

print(f"Created {len(chunks)} chunks from knowledge base")

# ============================================================================
# EMBEDDING AND VECTOR STORE
# ============================================================================
print("\n" + "=" * 70)
print("Step 3: Embedding and Vector Store")
print("=" * 70)

# Use free embeddings (or OpenAI for better quality)
use_openai_embeddings = os.getenv("OPENAI_API_KEY") is not None

if use_openai_embeddings:
    print("Using OpenAI embeddings")
    embeddings = OpenAIEmbeddings()
else:
    print("Using HuggingFace embeddings (free)")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# Create vector store
vectorstore = Chroma.from_texts(
    texts=chunks,
    embedding=embeddings,
    persist_directory="./kb_vectorstore"
)

print(f"Vector store created with {len(chunks)} chunks")
print("Saved to: ./kb_vectorstore")

# ============================================================================
# RETRIEVAL SETUP
# ============================================================================
print("\n" + "=" * 70)
print("Step 4: Retrieval Setup")
print("=" * 70)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}  # Top 3 relevant chunks
)

print("Retriever configured")

# ============================================================================
# CUSTOM PROMPT TEMPLATE
# ============================================================================
print("\n" + "=" * 70)
print("Step 5: Custom Prompt Template")
print("=" * 70)

prompt_template = """Use the following pieces of context to answer the question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}

Question: {question}

Answer: Provide a clear and helpful answer based on the context above."""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

print("Custom prompt template created")

# ============================================================================
# RAG CHAIN SETUP
# ============================================================================
print("\n" + "=" * 70)
print("Step 6: RAG Chain Setup")
print("=" * 70)

use_openai_llm = os.getenv("OPENAI_API_KEY") is not None

if use_openai_llm:
    print("Using OpenAI for generation")
    llm = OpenAI(temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    print("RAG chain ready with LLM")
else:
    print("OpenAI API key not found. Using retrieval-only mode.")
    print("Set OPENAI_API_KEY for full LLM integration")
    qa_chain = None

# ============================================================================
# QUESTION ANSWERING
# ============================================================================
print("\n" + "=" * 70)
print("Step 7: Question Answering")
print("=" * 70)

questions = [
    "How many vacation days do employees get?",
    "What is the IT support contact information?",
    "What are the key features of our product?",
    "What is the sales process?",
    "What are the company values?"
]

if qa_chain:
    print("\nAnswering questions with RAG...\n")
    for question in questions:
        print(f"{'='*70}")
        print(f"Q: {question}")
        print('='*70)
        
        result = qa_chain({"query": question})
        
        print(f"\nA: {result['result']}")
        print(f"\n📚 Sources ({len(result['source_documents'])}):")
        for i, doc in enumerate(result['source_documents'], 1):
            print(f"\n  Source {i}:")
            print(f"    {doc.page_content[:150]}...")
        print()
else:
    print("\nRetrieval-only mode:")
    for question in questions:
        print(f"\n{'='*70}")
        print(f"Q: {question}")
        print('='*70)
        
        docs = retriever.get_relevant_documents(question)
        print(f"\nRetrieved {len(docs)} relevant chunks:")
        for i, doc in enumerate(docs, 1):
            print(f"\n  Chunk {i}:")
            print(f"    {doc.page_content[:200]}...")
        print("\n[In full RAG: These would be passed to LLM for answer generation]")

# ============================================================================
# API FUNCTION
# ============================================================================
print("\n" + "=" * 70)
print("Step 8: API Function (For Deployment)")
print("=" * 70)

def ask_knowledge_base(question, qa_chain=None, retriever=None):
    """Ask a question to the knowledge base"""
    if qa_chain:
        result = qa_chain({"query": question})
        return {
            "answer": result['result'],
            "sources": [
                {
                    "content": doc.page_content[:200],
                    "metadata": doc.metadata if hasattr(doc, 'metadata') else {}
                }
                for doc in result['source_documents']
            ]
        }
    elif retriever:
        docs = retriever.get_relevant_documents(question)
        return {
            "answer": "Retrieval-only mode. Retrieved relevant information.",
            "sources": [doc.page_content for doc in docs]
        }
    else:
        return {"error": "Knowledge base not initialized"}

print("""
Example FastAPI integration:

from fastapi import FastAPI
app = FastAPI()

@app.post("/ask")
def ask(question: str):
    return ask_knowledge_base(question, qa_chain, retriever)
""")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"""
✅ Knowledge Base Assistant Built!
✅ Knowledge sections: {len(knowledge_base)}
✅ Chunks created: {len(chunks)}
✅ Vector store: ./kb_vectorstore
{'✅ LLM integration: Ready' if qa_chain else '⚠️  LLM integration: Set OPENAI_API_KEY'}

Next Steps:
1. Add more knowledge base content
2. Implement multi-turn conversations
3. Add user authentication
4. Deploy as API (FastAPI)
5. Create web interface
6. Add analytics and monitoring
7. Implement feedback loop
8. Scale to production
""")

print("=" * 70)

