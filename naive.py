from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


def naive(retriever,llm):

    prompt = PromptTemplate.from_template(""" Yor are a helpful assistant Context : {context} Question : {question} """)

    rag = ({"context": retriever,"question":RunnablePassthrough()} | prompt |llm| StrOutputParser())

    return rag