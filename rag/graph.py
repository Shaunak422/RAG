# GRAPH RAG — Entity-based retrieval with relationship synthesis.
# Extracts key entities (people, concepts, technologies) from the query,
# retrieves context for each entity independently, deduplicates across
# entities, reranks the combined pool, then asks the LLM to answer
# with awareness of relationships between the entities.
# Good for questions like "How does X relate to Y?"

from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from rag.reranker import rerank

MAX_ENTITIES = 5
MAX_CONTEXT_CHARS = 8000


def graph(retriever, llm):

    # Build chains once
    entity_chain = ChatPromptTemplate.from_template(
        """Extract key entities (concepts, people, technologies) from this query.
Query: {query}
Return as comma-separated list. Maximum 5. No generic words."""
    ) | llm | StrOutputParser()

    final_chain = ChatPromptTemplate.from_template(
        """Answer by identifying relationships between the entities.
Entities: {entities}
Context: {context}
Question: {query}
Answer:"""
    ) | llm | StrOutputParser()

    def _clean_entities(raw):
        # Filter out empty, short, and generic words
        stop = {"the", "this", "that", "it", "and", "or", "but", "a", "an", "is", "are", "to", "of", "in", "for"}
        entities = [e.strip().strip('"\'') for e in raw.split(",")]
        return [e for e in entities if len(e) >= 3 and e.lower() not in stop][:MAX_ENTITIES]

    def run(query):
        print(f"\n🔗 GRAPH RAG: '{query}'")

        # Extract entities
        raw = entity_chain.invoke({"query": query}).strip()
        entities = _clean_entities(raw)
        print(f"🏷️  Entities: {entities}")

        if not entities:
            entities = [query]  # fallback to full query

        # Retrieve per entity with dedup
        all_docs = []
        seen_keys = set()

        for entity in entities:
            docs = retriever.invoke(entity)
            for doc in docs:
                key = doc.page_content[:100]
                if key not in seen_keys:
                    seen_keys.add(key)
                    all_docs.append(doc)

        # Rerank combined pool against original query
        reranked = rerank(query, all_docs, top_n=5) if all_docs else []

        # Enforce token budget
        context = "\n\n".join([d.page_content for d in reranked])
        if len(context) > MAX_CONTEXT_CHARS:
            context = context[:MAX_CONTEXT_CHARS]

        return final_chain.invoke({"query": query, "context": context, "entities": ", ".join(entities)})

    return RunnableLambda(run)