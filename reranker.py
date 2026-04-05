from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


from sentence_transformers import CrossEncoder


re = CrossEncoder("BAAI/bge-reranker-base")

def rerank_docs(inputs):
        """
    inputs = {
        "question": query,
        "docs": retrieved documents
    }
    """
        
        query = inputs["question"]
        docs = inputs["docs"]

        pairs = [(query, d.page_content) for d in docs]

        s = re.predict(pairs)

        scored = list(zip(docs, s))

        scored.sort(key=lambda x: x[1], reverse=True)

        td = [doc.page_content for doc,_ in scored[:3]]

        return{"question": query, "context":"\n\n".join(td)}

def run(retriever,model):

    prompt = PromptTemplate.from_template(""" You are a helpful assistant context : {context} Question : {question} """)

    rag = ({"docs":retriever,"question":RunnablePassthrough()}| RunnableLambda(rerank_docs) | prompt | model | StrOutputParser())

    return rag

