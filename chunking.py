from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunks(documents, chunk_size=500, chunk_overlap=100):
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    return chunks