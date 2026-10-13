# AGENTIC RAG — Agent-driven retrieval loop with dynamic planning.
# Uses an LLM as a "planner" that decides what to do next after each
# retrieval step. The planner can see what context has been gathered
# so far, and decides whether to search for more info (with a new query)
# or STOP because enough context has been collected.
# Most powerful but slowest RAG type — best for exploratory questions.

from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from rag.reranker import rerank

MAX_CONTEXT_CHARS = 8000


def agentic(retriever, llm, max_steps=3):

    # Build chains once
    planner_chain = ChatPromptTemplate.from_template(
        """You are a retrieval agent. Decide the next action.
Original question: {original_query}
Current search: {current_query}
Context so far: {context_summary}
If context is SUFFICIENT, respond: STOP
If you need MORE info, respond with a NEW search query.
Respond with ONLY "STOP" or a new query."""
    ) | llm | StrOutputParser()

    final_chain = ChatPromptTemplate.from_template(
        """Answer using ALL gathered context.
Context: {context}
Question: {query}
Answer:"""
    ) | llm | StrOutputParser()

    def run(query):
        print(f"\n🤖 AGENTIC RAG: '{query}'")

        current_query = query
        context_memory = []
        seen_keys = set()
        seen_queries = set()

        for step in range(max_steps):
            print(f"\n📍 Step {step + 1}/{max_steps}: '{current_query}'")

            # Retrieve
            docs = retriever.invoke(current_query)
            seen_queries.add(current_query.lower())

            # Rerank
            reranked = rerank(current_query, docs, top_n=3) if docs else []

            # Dedup and add to memory
            for doc in reranked:
                key = doc.page_content[:100]
                if key not in seen_keys:
                    seen_keys.add(key)
                    context_memory.append(doc.page_content)

            # Ask planner what to do next
            summary = "\n---\n".join(context_memory[-3:]) or "(nothing yet)"
            action = planner_chain.invoke({
                "original_query": query,
                "current_query": current_query,
                "context_summary": summary
            }).strip()

            print(f"🧠 Planner: {action}")

            # Check stop conditions
            if "STOP" in action.upper():
                break
            if action.lower() in seen_queries:  # loop detection
                break
            if len(action.strip()) < 5:          # nonsense guard
                break

            current_query = action

        # Enforce token budget
        combined = "\n\n".join(context_memory)
        if len(combined) > MAX_CONTEXT_CHARS:
            combined = combined[:MAX_CONTEXT_CHARS]

        return final_chain.invoke({"query": query, "context": combined})

    return RunnableLambda(run)