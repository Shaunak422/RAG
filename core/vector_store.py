import os
from langchain_community.vectorstores import Chroma


def vector_store(chunks, embeddings, persist_directory='db'):
    # Load existing store if it exists, otherwise create new
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        vector = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

        # Get existing IDs to prevent duplicates
        existing = vector.get()
        existing_ids = set(existing.get("ids", []))

        # Filter to only new chunks
        new_chunks = [c for c in chunks if c.metadata.get("chunk_id", str(id(c))) not in existing_ids]

        if new_chunks:
            # Add only new chunks incrementally
            vector.add_documents(
                documents=new_chunks,
                ids=[c.metadata.get("chunk_id", str(id(c))) for c in new_chunks]
            )
            vector.persist()
            print(f"➕ Added {len(new_chunks)} new chunks (skipped {len(chunks) - len(new_chunks)} duplicates)")
        else:
            print("✅ No new chunks to add — store is up to date")

    else:
        # First run: create from scratch
        ids = [c.metadata.get("chunk_id", str(i)) for i, c in enumerate(chunks)]
        vector = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=persist_directory, ids=ids)
        vector.persist()
        print(f"🆕 Created new vector store with {len(chunks)} chunks")

    return vector
