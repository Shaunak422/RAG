from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def graph(retriever, llm):

    def run(query):

        # Step 1: extract entities
        entity_prompt = ChatPromptTemplate.from_template("""
        Extract key entities from the query.

        Query: {query}

        Return as comma-separated list.
        """)

        entity_chain = entity_prompt | llm | StrOutputParser()

        entities = entity_chain.invoke({"query": query}).strip()

        print("🔗 Entities:", entities)

        # Step 2: retrieve per entity
        entity_list = [e.strip() for e in entities.split(",")]

        all_context = []

        for e in entity_list:
            docs = retriever.invoke(e)

            context = "\n".join([d.page_content for d in docs])
            all_context.append(context)

        # Step 3: combine
        final_prompt = ChatPromptTemplate.from_template("""
        Answer using relationships between entities.

        Entities: {entities}

        Context:
        {context}

        Question: {query}
        """)

        final_chain = final_prompt | llm | StrOutputParser()

        return final_chain.invoke({
            "query": query,
            "context": "\n\n".join(all_context),
            "entities": entities
        })

    return RunnableLambda(run)