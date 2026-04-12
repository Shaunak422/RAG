from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def crag(retriever, llm):

    def run(query):

        print("\n🔎 ORIGINAL QUERY:", query)

        # Step 1: initial retrieval
        docs = retriever.invoke(query)

        context = "\n".join([d.page_content for d in docs])

        print(f"📄 Initial Docs Retrieved: {len(docs)}")

        # Step 2: evaluate retrieval quality
        eval_prompt = ChatPromptTemplate.from_template("""
        Evaluate if this context is sufficient to answer the question.

        Question: {query}

        Context:
        {context}

        Answer ONLY "GOOD" or "BAD".
        """)

        eval_chain = eval_prompt | llm | StrOutputParser()

        verdict = eval_chain.invoke({
            "query": query,
            "context": context
        }).strip().upper()

        print("🧠 Retrieval Quality:", verdict)

        # Step 3: if BAD → refine query
        if "BAD" in verdict:

            print("⚠️ Refining query...")

            refine_prompt = ChatPromptTemplate.from_template("""
            Rewrite this query to improve retrieval.

            Query: {query}
            """)

            refine_chain = refine_prompt | llm | StrOutputParser()

            new_query = refine_chain.invoke({"query": query}).strip()

            print("🔁 New Query:", new_query)

            docs = retriever.invoke(new_query)

            context = "\n".join([d.page_content for d in docs])

            print(f"📄 New Docs Retrieved: {len(docs)}")

        # Step 4: final answer
        final_prompt = ChatPromptTemplate.from_template("""
        Answer the question using this context:

        {context}

        Question: {query}
        """)

        final_chain = final_prompt | llm | StrOutputParser()

        return final_chain.invoke({
            "query": query,
            "context": context
        })

    return RunnableLambda(run)