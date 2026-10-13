# NAIVE RAG — Simplest RAG pipeline.
# Retrieves relevant chunks and passes them directly to the LLM.
# No reranking, no correction, no multi-step — just retrieve and answer.
# Best used as a baseline to compare other RAG types against.

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


def naive(retriever, llm):

    prompt = PromptTemplate.from_template(
        """You are a helpful and accurate assistant. Answer the question 
based ONLY on the provided context. If the context does not contain 
enough information, say "I don't have enough information."

Context:
{context}

Question: {question}

Answer:"""
    )

    # Retriever fetches docs → prompt formats them → LLM answers → parse output
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain