"""
RAG System — Main Application Entry Point

Pipeline Flow:
    1. INGESTION  → Load PDFs from /data (skip already-ingested via hash)
    2. EMBEDDING  → Load embedding model (singleton, once per session)
    3. CHUNKING   → Split new docs into semantic chunks with overlap
    4. VECTOR DB  → Store chunks incrementally (skip existing chunk IDs)
    5. RETRIEVER  → Create MMR retriever (diverse top-k candidates)
    6. LLM        → Load Llama 3.3 70B via Groq API
    7. RAG CHAIN  → Build selected RAG pipeline
    8. QUERY LOOP → Interactive Q&A until user types 'exit'

RAG Modes (selectable via CLI argument):
    python app.py naive      → Basic retrieval + answer
    python app.py reranker   → Cross-encoder reranking
    python app.py crag       → Self-correcting retrieval (default)
    python app.py multihop   → Query decomposition
    python app.py graph      → Entity-based retrieval
    python app.py agentic    → Agent-driven retrieval loop

Incremental Processing:
    - First run: processes all PDFs, creates vector store from scratch
    - Subsequent runs: only processes new/changed PDFs (hash-based dedup)
    - If no new files: skips ingestion entirely, loads existing vector store
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables (.env file contains GROQ_API_KEY)
load_dotenv()

# -- Core pipeline modules --
from core.ingestion import load        # PDF loading with hash dedup
from core.chunking import chunks       # Semantic paragraph-based chunking
from core.embeddings import embeddings # Singleton embedding model
from core.vector_store import vector_store  # Incremental ChromaDB storage
from core.retriever import retriever   # MMR diversity retriever
from core.llm import llm              # Groq LLM (Llama 3.3 70B)

# -- RAG strategy modules --
from rag.naive import naive            # Basic RAG (baseline)
from rag.reranker import run           # Cross-encoder reranking
from rag.multihop import multihop      # Query decomposition
from rag.crag import crag             # Corrective RAG
from rag.graph import graph           # Entity-based retrieval
from rag.agentic import agentic       # Agent-driven retrieval

# -- Available RAG modes --
# Maps CLI argument names to their constructor functions.
# All constructors share the same signature: fn(retriever, llm) -> chain
RAG_MODES = {
    "naive": naive,
    "reranker": run,
    "crag": crag,
    "multihop": multihop,
    "graph": graph,
    "agentic": agentic,
}


def main():
    """
    Main application entry point.

    Executes the full RAG pipeline:
        1. Parse CLI args for RAG mode selection
        2. Ingest new PDFs (skip unchanged files)
        3. Chunk and embed new documents
        4. Update vector store incrementally
        5. Create retriever and LLM
        6. Build RAG chain
        7. Run interactive Q&A loop
    """

    # ================================================================
    # Step 0: Parse CLI arguments
    # =======================================================   =========
    # Usage: python app.py <rag_mode>
    # Default: crag (Corrective RAG — best balance of quality and robustness)
    rag_mode = "crag"
    if len(sys.argv) > 1 and sys.argv[1] in RAG_MODES:
        rag_mode = sys.argv[1]

    print(f"🧠 RAG Mode: {rag_mode}")
    print(f"   Available modes: {', '.join(RAG_MODES.keys())}")

    # ================================================================
    # Step 1: Ingestion — Load only NEW PDFs
    # ================================================================
    # The load() function:
    #   - Walks /data directory for .pdf files
    #   - Computes SHA-256 hash of each file
    #   - Skips files whose hash is in db/.ingested_hashes.json
    #   - Returns only new/changed documents
    print("\n" + "=" * 60)
    print("📥 Step 1: Ingestion")
    print("=" * 60)
    docs = load()

    # ================================================================
    # Step 2: Embedding model — Load once (singleton)
    # ================================================================
    # The embeddings() function returns a singleton HuggingFaceEmbeddings
    # instance. Called once here, reused by vector_store internally.
    print("\n" + "=" * 60)
    print("🔢 Step 2: Loading embedding model")
    print("=" * 60)
    embed = embeddings()

    # ================================================================
    # Step 3 & 4: Chunk + Store (only if new documents exist)
    # ================================================================
    if docs:
        # New files were found — process them
        pdf_names = list(set([doc.metadata['source'] for doc in docs]))
        print(f"\n📄 New files detected: {pdf_names}")

        # Step 3: Chunk new documents
        # Semantic paragraph-based splitting with 15% overlap
        print("\n" + "=" * 60)
        print("✂️  Step 3: Chunking new documents")
        print("=" * 60)
        chunk = chunks(docs)
        print(f"Chunks created: {len(chunk)}")

        # Step 4: Update vector store incrementally
        # Only adds chunks with new chunk_ids (skips existing ones)
        print("\n" + "=" * 60)
        print("💾 Step 4: Updating vector store")
        print("=" * 60)
        vector = vector_store(chunk, embed, persist_directory='db')
    else:
        # No new files — load existing vector store directly
        # This is the fast path for subsequent runs with no new data
        print("\n✅ No new files — loading existing vector store")
        from langchain_community.vectorstores import Chroma

        if os.path.exists('db') and os.listdir('db'):
            vector = Chroma(persist_directory='db', embedding_function=embed)
            print(f"📂 Loaded existing vector store from 'db/'")
        else:
            print("❌ No vector store found and no documents to ingest!")
            print("   Please add PDF files to the /data folder and run again.")
            return

    # ================================================================
    # Step 5: Create Retriever
    # ================================================================
    # MMR (Maximal Marginal Relevance) retriever with:
    #   - k=10: retrieve 10 diverse candidates (for downstream reranking)
    #   - lambda_mult=0.7: balance relevance vs diversity
    print("\n" + "=" * 60)
    print("🔍 Step 5: Creating retriever (MMR, k=10)")
    print("=" * 60)
    retr = retriever(vector)

    # ================================================================
    # Step 6: Load LLM
    # ================================================================
    # ChatGroq with Llama 3.3 70B Versatile
    # Requires GROQ_API_KEY in .env
    print("\n" + "=" * 60)
    print("🤖 Step 6: Loading LLM")
    print("=" * 60)
    model = llm()
    print("✅ LLM ready (Llama 3.3 70B via Groq)")

    # ================================================================
    # Step 7: Build RAG Chain
    # ================================================================
    # Construct the selected RAG pipeline
    rag_fn = RAG_MODES[rag_mode]
    rag_chain = rag_fn(retr, model)
    print(f"\n✅ RAG chain built: {rag_mode}")

    # ================================================================
    # List available documents
    # ================================================================
    # Query the vector store metadata to show what PDFs are indexed
    try:
        all_meta = vector.get()
        sources = sorted(list(set(
            m.get("source", "unknown")
            for m in all_meta.get("metadatas", [])
            if m
        )))
    except Exception:
        sources = []

    print(f"\n📚 Available PDFs in vector store: {len(sources)}")
    for s in sources:
        print(f"  📄 {s}")

    # ================================================================
    # Step 8: Interactive Query Loop
    # ================================================================
    print("\n" + "=" * 60)
    print("💬 Ready for questions! Type 'exit' to quit.")
    print("=" * 60)

    while True:
        query = input("\n❓ Ask question: ")

        # Exit condition
        if query.lower().strip() == "exit":
            print("👋 Goodbye!")
            break

        # Skip empty queries
        if not query.strip():
            print("⚠️  Please enter a question.")
            continue

        try:
            # Invoke the RAG chain with the user's query
            ans = rag_chain.invoke(query)
            print(f"\n✅ Answer:\n{ans}")

        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
