#!/usr/bin/env python3

# Sample code to implement a Llama2 chatbot with Langchain and LlamaCpp

from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from sys import argv, exit
from os import path

# Get the path to the LLM model we are using.
if (len(argv) != 2):
    print("Usage:", argv[0], "/path/to/llm")
    exit(1)
llama_path = argv[1]
if not path.exists(llama_path):
    print("Invalid llama model path!")
    exit(1)

# Load the LlamaCpp language model and adjust GPU usage
llm = LlamaCpp(
    model_path=llama_path,
    n_ctx=2048,
    n_gpu_layers=-1,
    n_batch=512,
    verbose=False,
)

# Define the prompt template
template = """
Question: {question}
Answer:
"""
prompt = PromptTemplate(template=template, input_variables=["question"])

# Create an LLMChain to manage interactions with the prompt and model
llm_chain = LLMChain(prompt=prompt, llm=llm)

print("Chatbot initialized, ready to chat...")
while True:
    question = input("> ")
    answer = llm_chain.invoke(question)['text']
    print(answer, '\n')
