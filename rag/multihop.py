# MULTIHOP RAG — Decomposes complex questions into sub-questions.
# For questions like "Compare X and Y", a single retrieval might only
# find docs about X. Multihop breaks it into sub-questions ("What is X?",
# "What is Y?", "How do they differ?"), retrieves for each independently,
# deduplicates the results, and synthesizes a comprehensive answer.

from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from rag.reranker import rerank

MAX_CONTEXT_CHARS = 8000  # token budget to prevent context overflow


def multihop(retriever, llm, max_steps=3):

    # Build chains once
    decompose_chain = ChatPromptTemplate.from_template(
        """Break this question into {max_steps} smaller sub-questions.
Question: {query}
Return each on a new line, no numbering."""
    ) | llm | StrOutputParser()

    final_chain = ChatPromptTemplate.from_template(
        """Answer using ALL the context gathered from multiple sub-queries.
Question: {query}
Context: {context}
Answer:"""
    ) | llm | StrOutputParser()

    def _parse_sub_questions(raw):
        # Clean LLM output into list of sub-questions
        lines = raw.strip().split("\n")
        return [l.strip().lstrip("0123456789.-)*• ").strip() for l in lines if len(l.strip()) > 10]

    def run(query):
        print(f"\n🔗 MULTIHOP: '{query}'")

        # Decompose into sub-questions
        raw = decompose_chain.invoke({"query": query, "max_steps": max_steps})
        sub_questions = _parse_sub_questions(raw)
        print(f"🔍 Sub-questions: {len(sub_questions)}")

        seen_keys = set()
        all_context = []

        for i, sub_q in enumerate(sub_questions):
            print(f"  {i+1}. {sub_q}")
            docs = retriever.invoke(sub_q)

            if not docs:
                continue

            # Rerank per sub-question
            reranked = rerank(sub_q, docs, top_n=3)

            # Dedup across sub-questions
            for doc in reranked:
                key = doc.page_content[:100]
                if key not in seen_keys:
                    seen_keys.add(key)
                    all_context.append(doc.page_content)

        # Enforce token budget
        combined = "\n\n".join(all_context)
        if len(combined) > MAX_CONTEXT_CHARS:
            combined = combined[:MAX_CONTEXT_CHARS]

        return final_chain.invoke({"query": query, "context": combined})

    return RunnableLambda(run)