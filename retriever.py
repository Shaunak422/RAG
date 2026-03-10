def retriever(vector, k=3, source=None):

    if source:
        r = vector.as_retriever(
            search_kwargs={
                "k": k,
                "filter": {"source": source}
            }
        )
    else:
        r = vector.as_retriever(
            search_kwargs={"k": k}
        )

    return r