import streamlit as st
import openai
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS, Pinecone
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import pinecone

LETTER_TEMPLATE = """ You are tasked with retrieving questions regarding generic knowledge documents.

Provide an answer based on this retreival, and if you can't find anything relevant, just say "I'm sorry, I couldn't find that."

{context}

Question: {question}
Answer:
"""
LETTER_PROMPT = PromptTemplate(input_variables=["question", "context"], template=LETTER_TEMPLATE, )

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.1,
    max_tokens=1000, 
    openai_api_key=st.secrets["openai_key"]
)


def get_pinecone():
    " get the pinecone embeddings"
    pinecone.init(
        api_key=st.secrets['pinecone_key'], 
        environment=st.secrets['pinecone_env'] 
        )
    
    index_name = "salesgpt"
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["openai_key"])
    return Pinecone.from_existing_index(index_name,embeddings)


def letter_chain(question):
    """returns a question answer chain for pinecone vectordb"""
    
    docsearch = get_pinecone()
    retreiver = docsearch.as_retriever(#
        #search_type="similarity", #"similarity", "mmr"
        search_kwargs={"k":3}
    )
    qa_chain = RetrievalQA.from_chain_type(llm, 
                                            retriever=retreiver,
                                           chain_type="stuff", #"stuff", "map_reduce","refine", "map_rerank"
                                           return_source_documents=True,
                                           #chain_type_kwargs={"prompt": LETTER_PROMPT}
                                          )
    return qa_chain({"query": question})


# def letter_qa(query, temperature=.1,model_name="gpt-3.5-turbo"):
#     """
#     this method was deprecated but seems to be more efficient from a token perspective
#     """
#     pdf_qa = ChatVectorDBChain.from_llm(OpenAI(temperature=temperature, model_name=model_name, openai_api_key=st.secrets["openai_key"]),
#                     pinecone_search(), return_source_documents=True)
#     return pdf_qa({"question": query, "chat_history": ""})

