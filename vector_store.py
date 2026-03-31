from langchain_community.vectorstores import Chroma

def vector_store(chunks,embeddings,persist_directory='db'):
    
    vector = Chroma.from_documents(
        documents = chunks,
        embedding = embeddings,
        persist_directory = persist_directory
    )

    vector.persist()
    return vector

