#!/usr/bin/env python3

# Sample code to implement a Llama2 chatbot with RAG using a Chroma vector
# database with Langchain and LlamaCpp

from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from sys import argv, exit
from os import path

# Get the path to the LLM model we are using, and the path to chromadb for RAG.
if (len(argv) != 3):
    print("Usage:", argv[0], "/path/to/llm /path/to/chromadb/dir")
    exit(1)
llama_path = argv[1]
chromadbdir = argv[2]
chromadbdir = chromadbdir + "/chromadb"
if not path.exists(llama_path):
    print("Invalid llama model path!")
    exit(1)
if not path.exists(chromadbdir):
    print("Invalid chroma database path!")
    exit(1)

# Define the prompt template 
# A context must be passed in to the RetrievalQA even if it is empty.
template = """
Context: {context}
Question: {question}
Answer:
"""
prompt = PromptTemplate(
    template=template,
    input_variables=['context', 'question']
)

# Load Chroma vector database from disk
embedding_function = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2"
)
db = Chroma(persist_directory=chromadbdir,
    embedding_function=embedding_function
)

# Load the LlamaCpp language model and adjust GPU usage
llm = LlamaCpp(
    model_path=llama_path,
    n_ctx=2048,
    n_gpu_layers=-1,
    n_batch=512,
    verbose=False,
)

# Setup the RAG chain with our LLM
retriever = db.as_retriever(search_kwargs={'k': 2})
chain = RetrievalQA.from_chain_type(llm=llm,
                                    chain_type='stuff',
                                    retriever=retriever,
                                    return_source_documents=False,
                                    chain_type_kwargs={'prompt': prompt})
llm_chain = LLMChain(prompt=prompt, llm=llm)

print("Chatbot initialized, ready to chat...")
while True:
    question = input("> ")
    answer = chain({'query':question})['result']
    print(answer, '\n')
