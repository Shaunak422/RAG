# CRAG (Corrective RAG) — Self-correcting retrieval pipeline.
# After retrieving docs, it asks the LLM to evaluate if the context is
# good enough to answer the question. If the LLM says "BAD", it rewrites
# the query and retrieves again. Repeats up to max_retries times.
# This fixes cases where the user's wording doesn't match the document's
# terminology, leading to poor initial retrieval.

from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from rag.reranker import rerank


def crag(retriever, llm, max_retries=2):

    # Build chains once (reused across queries)
    eval_chain = ChatPromptTemplate.from_template(
        """Evaluate if this context can answer the question.
Question: {query}
Context: {context}
Answer ONLY "GOOD" or "BAD"."""
    ) | llm | StrOutputParser()

    refine_chain = ChatPromptTemplate.from_template(
        """Rewrite this query to improve retrieval results.
Original query: {query}
Return ONLY the rewritten query."""
    ) | llm | StrOutputParser()

    final_chain = ChatPromptTemplate.from_template(
        """Answer using ONLY the provided context.
Context: {context}
Question: {query}
Answer:"""
    ) | llm | StrOutputParser()

    def run(query):
        print(f"\n🔎 CRAG: '{query}'")
        current_query = query
        best_context = ""

        for attempt in range(1 + max_retries):
            # Retrieve docs
            docs = retriever.invoke(current_query)
            print(f"📄 Attempt {attempt + 1}: Retrieved {len(docs)} docs")

            if not docs:
                if attempt < max_retries:
                    current_query = refine_chain.invoke({"query": current_query}).strip()
                    continue
                break

            # Rerank to get best candidates
            reranked = rerank(current_query, docs, top_n=5)

            # Deduplicate overlapping chunks
            seen = set()
            unique = []
            for doc in reranked:
                key = doc.page_content[:100]
                if key not in seen:
                    seen.add(key)
                    unique.append(doc.page_content)
            best_context = "\n\n".join(unique)

            # Evaluate quality
            verdict = eval_chain.invoke({"query": query, "context": best_context}).strip().upper()
            print(f"🧠 Verdict: {verdict}")

            if "GOOD" in verdict:
                break

            # Refine query if BAD and retries left
            if attempt < max_retries:
                current_query = refine_chain.invoke({"query": current_query}).strip()
                print(f"🔁 Refined: '{current_query}'")

        return final_chain.invoke({"query": query, "context": best_context})

    return RunnableLambda(run)