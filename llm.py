import os
from langchain_groq import ChatGroq

def llm():
    return ChatGroq(model="llama-3.3-70b-versatile",temperature=0.5)

