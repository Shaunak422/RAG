from difflib import SequenceMatcher


def retriever(vector, k=10, source=None, use_mmr=True):
    # Create retriever with MMR for diverse results
    search_type = "mmr" if use_mmr else "similarity"
    search_kwargs = {"k": k}

    if use_mmr:
        search_kwargs["fetch_k"] = k * 3      # fetch more candidates for MMR selection
        search_kwargs["lambda_mult"] = 0.7     # balance relevance vs diversity

    if source:
        search_kwargs["filter"] = {"source": source}

    r = vector.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
    return r


def deduplicate_chunks(docs, similarity_threshold=0.9):
    # Remove near-duplicate chunks based on content overlap
    unique = []
    for doc in docs:
        is_dup = False
        for existing in unique:
            ratio = SequenceMatcher(None, doc.page_content, existing.page_content).ratio()
            if ratio >= similarity_threshold:
                is_dup = True
                break
        if not is_dup:
            unique.append(doc)
    return unique