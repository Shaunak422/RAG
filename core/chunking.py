from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunks(documents, chunk_size=800, chunk_overlap=120):
    # Semantic splitting: paragraph → sentence → word boundaries
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )

    # Split all documents into chunks
    split_chunks = splitter.split_documents(documents)

    # Add unique chunk_id to each chunk for traceability
    for i, chunk in enumerate(split_chunks):
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", 0)
        chunk.metadata["chunk_id"] = f"{source}_p{page}_c{i}"

    return split_chunks