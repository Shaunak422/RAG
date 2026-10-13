import hashlib
from langchain_community.embeddings import HuggingFaceEmbeddings

_embedding_model = None  # singleton instance
_embedding_cache = {}    # hash → embedding vector cache


def embeddings():
    # Return singleton embedding model with L2 normalization
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en",
            encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
        )
    return _embedding_model


def get_cached_embeddings(texts):
    # Batch embed texts, skip already-cached ones
    model = embeddings()
    results = [None] * len(texts)
    texts_to_embed = []
    indices_to_embed = []

    for i, text in enumerate(texts):
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in _embedding_cache:
            results[i] = _embedding_cache[text_hash]
        else:
            texts_to_embed.append(text)
            indices_to_embed.append(i)

    # Only embed uncached texts
    if texts_to_embed:
        new_embeddings = model.embed_documents(texts_to_embed)
        for idx, emb, text in zip(indices_to_embed, new_embeddings, texts_to_embed):
            text_hash = hashlib.md5(text.encode()).hexdigest()
            _embedding_cache[text_hash] = emb
            results[idx] = emb

    return results