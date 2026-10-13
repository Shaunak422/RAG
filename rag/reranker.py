# RERANKER RAG — Two-stage retrieval with cross-encoder reranking.
# Stage 1: Bi-encoder retriever fetches top-k candidates (fast, approximate).
# Stage 2: Cross-encoder reranker scores each (query, doc) pair jointly (slow, accurate).
# The cross-encoder sees query + doc together, making it much more precise than
# cosine similarity. But it's too slow for full-corpus search, so we only
# rerank the top-k candidates from stage 1.
# Also exports a standalone rerank() function usable by other RAG types.

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder

_reranker_model = None  # lazy-loaded singleton


def _get_reranker():
    # Load cross-encoder model only when first needed
    global _reranker_model
    if _reranker_model is None:
        _reranker_model = CrossEncoder("BAAI/bge-reranker-base")
    return _reranker_model


def rerank(query, docs, top_n=5):
    # Standalone reranker — can be imported by any RAG type
    if not docs:
        return []

    model = _get_reranker()
    pairs = [(query, doc.page_content) for doc in docs]  # create query-doc pairs
    scores = model.predict(pairs)                         # score all pairs in batch
    scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)  # sort by score
    return [doc for doc, score in scored_docs[:top_n]]    # return top_n


def rerank_docs(inputs):
    # LCEL-compatible wrapper for chain pipelines
    query = inputs["question"]
    docs = inputs["docs"]
    top_docs = rerank(query, docs, top_n=3)
    context = "\n\n".join([doc.page_content for doc in top_docs])
    return {"question": query, "context": context}


def run(retriever, model):

    prompt = PromptTemplate.from_template(
        """You are a helpful and accurate assistant. Answer based ONLY on the context.

Context:
{context}

Question: {question}

Answer:"""
    )

    # Retrieve → rerank with cross-encoder → prompt → LLM → parse
    rag_chain = (
        {"docs": retriever, "question": RunnablePassthrough()}
        | RunnableLambda(rerank_docs)
        | prompt
        | model
        | StrOutputParser()
    )

    return rag_chain
