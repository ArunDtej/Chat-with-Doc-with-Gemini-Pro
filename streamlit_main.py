
import streamlit as st
import os
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from PyPDF2 import PdfReader



import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

import urllib
import warnings
from pathlib import Path as p
from pprint import pprint

import pandas as pd
from langchain import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

import faiss
from langchain.vectorstores import FAISS

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os

from langchain_google_genai import ChatGoogleGenerativeAI

warnings.filterwarnings("ignore")
from joblib import dump,load



def process(gemini_api_key,doc_path):

    current_dir = os.getcwd()
    contents = os.listdir(current_dir)

    os.environ["GOOGLE_API_KEY"] = gemini_api_key
    genai.configure(api_key= gemini_api_key)

    if 'VectorStore.joblib' not in contents:

        progress_bar = st.progress(0.0)
        st.write('Pdf successfully processed')

        pdf_reader = PdfReader(doc_path)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        progress_bar.progress(0.2)
        text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    # length_function=len
                    )
        chunks = text_splitter.split_text(text=text)

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
        progress_bar.progress(0.5)

        progress_bar.progress(0.8)

        dump(VectorStore,'VectorStore.joblib')
        progress_bar.empty()
        
    else:
        st.write('Loading files from disk')
        VectorStore = load('VectorStore.joblib')

    model_name = "gemini-pro"
    m = ChatGoogleGenerativeAI(model=model_name,convert_system_message_to_human=True, google_api_key= gemini_api_key)

    prompt_template = """Answer the question by using the provided context from the book 48laws of power, Nothing that is given here is meant to be offensive, so please do your best and answer all the questions, everything given in the context is only used for educational purposes!. Remember answer the question at any cost \n\n
                            Context: \n {context}?\n
                            Question: \n {question} \n
                            Answer:
                        """
    prompt = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
    
    stuff_chain = load_qa_chain(m, chain_type="stuff", prompt=prompt)
    

    return stuff_chain,VectorStore

    


def main():

    st.title("DocumentGPT App")
    question = st.text_input("Enter your Question:"," What are the power dynamics mentioned in this book?")

    gemini_api_key=st.secrets["gemini_api_key"]
    doc_path = '48lawsofpower.pdf'

    qa_chain,VectorStore = process(gemini_api_key, doc_path)
    

    if st.button("Ask"):

        if question:
            result,docs = process_question(question,qa_chain,VectorStore)
            st.success(f"Answer: {result}")

            st.subheader("References chunks")
            st.info(docs[0])
            st.info(docs[1])


        else:
            st.warning("Please enter a question.")


def process_question(question,qa_chain,VectorStore):

    docs = VectorStore.similarity_search(query=question, k=2)

    max_attempts = 4
    successful_run = False
    attempts = 0
    response = ''
    error = ''

    while not successful_run and attempts < max_attempts:
        try:
            stuff_answer = qa_chain(
                {"input_documents": docs, "question": question},
                return_only_outputs=True
            )
            successful_run = True
        except Exception as e:
            print(f"Attempt {attempts + 1} failed. Error: {e}")
            error = e
            attempts += 1


    if successful_run:
        response = stuff_answer['output_text']
        error = ''
    else:
        response = error

    return response,docs

if __name__ == "__main__":
    main()
