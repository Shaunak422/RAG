from langchain_community.embeddings import HuggingFaceEmbeddings

def embeddings():
        return HuggingFaceEmbeddings(
                model_name="BAAI/bge-base-en"
    )