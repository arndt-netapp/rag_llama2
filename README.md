Sample code to implement a Llama2 chatbot with Langchain and LlamaCpp. as well
as a Llama2 chatbot with RAG using a Chroma vector database.  

This is meant to be used with Python3 in a venv, with all required libraries
installed via pip.

```
arndt@rag:~$ python3 -m venv rag
arndt@rag:~$ . rag/bin/activate
(rag) arndt@rag:~$ 
pip install llama-cpp-python
pip install langchain
pip install sentence-transformers
pip install chromadb
pip install chardet
pip install netapp-dataops-traditional
```

The llama-2-7b-chat.Q4_K_M.gguf model will run on CPU and 8GB of ram, but will
run faster on GPU :).  Note the following configuration settings and
llama-cpp-python installation comamands to get GPU support.  This is to be done
after the CUDA libraries are installed, which match the driver details as found
in the "nvidia-smi" command output.

```
export PATH="${PATH}:/usr/local/cuda/bin"
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir
```
