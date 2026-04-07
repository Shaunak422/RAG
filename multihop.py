# from langchain_core.runnables import RunnableLambda

# def multihop(retriever, llm, max_hops=3):

#     def run_multihop(query):

#         from langchain_core.prompts import ChatPromptTemplate
#         from langchain_core.output_parsers import StrOutputParser

#         reasoning_prompt = ChatPromptTemplate.from_template("""
#         Question: {query}

#         Context:
#         {context}

#         Generate a SHORT search query (max 10 words).
#         Only return the query.
#         """)

#         reasoning_chain = reasoning_prompt | llm | StrOutputParser()

#         final_prompt = ChatPromptTemplate.from_template("""
#         Answer the question:

#         {query}

#         Using this context:
#         {context}
#         """)

#         final_chain = final_prompt | llm | StrOutputParser()

#         current_query = query
#         all_context = []
#         seen_queries = set()

#         for hop in range(max_hops):

#             print(f"\n================ HOP {hop+1} ================")
#             print(f"🔍 Current Query: {current_query}")
#             docs = retriever.invoke(current_query)

#             if not docs:
#                 break

#             context = "\n".join([doc.page_content for doc in docs])
#             all_context.append(context)

#             next_query = reasoning_chain.invoke({
#                 "query": current_query,
#                 "context": context
#             }).strip()

#             if (
#                 next_query.lower() == current_query.lower()
#                 or next_query in seen_queries
#                 or len(next_query) < 5
#             ):
#                 break

#             seen_queries.add(current_query)
#             current_query = next_query

#         final_context = "\n\n".join(all_context)

#         return final_chain.invoke({
#             "query": query,
#             "context": final_context
#         })

#     # 🔥 THIS MAKES IT LCEL
#     return RunnableLambda(run_multihop)




from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def multihop(retriever, llm, max_steps=3):

    def run(query):

        # Step 1: Decompose query
        decompose_prompt = ChatPromptTemplate.from_template("""
        Break this question into {max_steps} smaller sub-questions.

        Question: {query}

        Return as a list.
        """)

        decompose_chain = decompose_prompt | llm | StrOutputParser()

        sub_questions = decompose_chain.invoke({
            "query": query,
            "max_steps": max_steps
        }).split("\n")

        all_context = []

        print("\n🔍 Decomposed Questions:")
        for i, q in enumerate(sub_questions):
            print(f"{i+1}. {q}")

            docs = retriever.invoke(q)
            context = "\n".join([d.page_content for d in docs])
            all_context.append(context)

        # Final answer
        final_prompt = ChatPromptTemplate.from_template("""
        Answer the question using all context.

        Question: {query}
        Context:
        {context}
        """)

        final_chain = final_prompt | llm | StrOutputParser()

        return final_chain.invoke({
            "query": query,
            "context": "\n\n".join(all_context)
        })

    return RunnableLambda(run)