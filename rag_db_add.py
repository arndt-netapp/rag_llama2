#!/usr/bin/env python3

# Sample code to create or update a Chroma vector database from unstructured
# data.  In this sample we add to Chroma from a directory path of text files.

# Import libraries
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_community.vectorstores import Chroma
from sys import argv, exit
from os import path

# Get the directory that we will recursively scan for files to add to Chroma,
# as well as the path to where we should create or update the Chroma DB.
if (len(argv) != 3):
    print("Usage:", argv[0], "/path/to/files /path/to/chromadb")
    exit(1)
datadir = argv[1]
chromadbdir = argv[2]
if not path.exists(datadir):
    print("Invalid data input path!")
    exit(1)
if not path.exists(chromadbdir):
    print("Invalid chroma database path!")
    exit(1)
chromadbdir = chromadbdir + "/chromadb"

# Load our documents
text_loader_kwargs={'autodetect_encoding': True}
loader = DirectoryLoader(datadir, glob="**/*.txt", use_multithreading=True,
    loader_cls=TextLoader, loader_kwargs=text_loader_kwargs, recursive=True,
    show_progress=False, 
)
documents = loader.load()

# Split the documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=150,
    separator="\n",
)
docs = text_splitter.split_documents(documents)

# Initialize ChromaDB and insert the documents
embedding_function = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2"
)
db = Chroma.from_documents(docs, embedding_function,
    persist_directory=chromadbdir
)
db.persist()

# Each document gets stored as multiple chunks, let's see how many we have.
chunks = db.get(include=['metadatas'])
dbsize = len(chunks['ids'])
print("There are now", dbsize, "document chunks in the db.")
