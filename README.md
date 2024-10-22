# 📄 Document question answering using Groq API

A simple Streamlit app that answers questions about an uploaded document via Groq API.

### How to run it

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```

This readme describes a system for question answering over a uploaded PDF document using Groq's API and Streamlit.

### Prerequisites
1. Groq API Key: You will need a Groq API key to access their Large Language Models (LLMs).

### Functionality
1. Input Groq API Key: The system will prompt you to enter your Groq API key securely.
2. Upload PDF Document: You can upload a PDF document for which you want to answer questions.
3. Document Vectorization: The system will process the uploaded PDF and convert it into a vector representation suitable for Groq's LLMs.
4. Question Answering: You can then ask questions related to the document. The system will use Groq's API and your chosen LLM model to answer your questions based on the document content.

# Building a Document QA Application with Streamlit and Groq

In the world of natural language processing, efficiently extracting information from documents is a vital task. Today, we'll explore how to build a Document Question Answering (QA) application using Python, Streamlit, and Groq's cutting-edge AI models. This application allows users to upload a PDF document, analyze it, and ask questions about its content.

## Key Features

1. **Effortless Document Analysis**: Upload a PDF document and transform it into a searchable vector database.
2. **Interactive Model Selection**: Choose from an array of Groq models optimized for various use-cases.
3. **Seamless Question Answering**: Input your query and receive precise responses based on the document's content.

## Technologies Used

- **Streamlit**: A powerful open-source app framework for creating interactive web applications in Python.
- **Langchain**: A suite of tools for managing language processing tasks, providing connectors to language models, and handling document parsing.
- **Groq Models**: Advanced AI models for natural language processing.

## Step-by-Step Guide

### 1. Setting Up the Project

Start by importing the necessary modules. Make sure you have Streamlit and all relevant Langchain libraries installed in your environment.

```python
import streamlit as st
import tempfile
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
```

### 2. Building the Streamlit Interface

The frontend of the application allows users to interact with the system. The interface will let users upload a PDF, select a model, and input their Groq API key.

```python
st.title("📄 Document Question Answering")
st.write("Upload a document below and ask a question about it – Groq will answer!")

# Sidebar for model and API key input
with st.sidebar:
    model_options = [
        "llama3-8b-8192", "llama3-70b-8192", "llama-3.1-8b-instant", 
        "llama-3.1-70b-versatile", "llama-3.2-1b-preview", "llama-3.2-3b-preview",
        "llama-3.2-11b-text-preview", "llama-3.2-90b-text-preview",
        "mixtral-8x7b-32768", "gemma-7b-it", "gemma2-9b-it"
    ]
    selected_model = st.selectbox("Select any Groq Model", model_options)
    groq_api_key = st.text_input("Groq API Key", type="password")
```

### 3. Handling Document Upload and Processing

Once a PDF is uploaded, we split its text into manageable chunks and convert it into a vectorized format using FAISS. This step allows the Groq models to efficiently retrieve and process information.

```python
def create_vector_db(pdf_file):
    if "vector_store" not in st.session_state:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(pdf_file.read())
            pdf_file_path = temp_file.name
        
        st.session_state.embeddings = HuggingFaceEmbeddings(
            model_name='BAAI/bge-small-en-v1.5', 
            model_kwargs={'device': 'cpu'}, 
            encode_kwargs={'normalize_embeddings': True}
        )
        
        loader = PyPDFLoader(pdf_file_path)
        text_document = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        document_chunks = text_splitter.split_documents(text_document)
        st.session_state.vector_store = FAISS.from_documents(document_chunks, st.session_state.embeddings)
        
pdf_input = st.file_uploader("Upload the PDF file", type=['pdf'])
if pdf_input and st.button("Create the Vector DB from PDF"):
    create_vector_db(pdf_input)
    st.success("Vector Store DB for this PDF file is ready")
```

### 4. Querying and Displaying Results

Users can now input questions to query the document. The application uses a Groq model to generate responses based on the uploaded document's content.

```python
if "vector_store" in st.session_state:
    user_prompt = st.text_input("Enter Your Question related to the uploaded PDF")
    if st.button('Submit Prompt') and user_prompt:
        llm = ChatGroq(groq_api_key=groq_api_key, model_name=selected_model)
        prompt = ChatPromptTemplate.from_template(
            """
            Answer the questions based on the provided context only.
            <context>
            {context}
            <context>
            Questions: {input}
            """
        )
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vector_store.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        response = retrieval_chain.invoke({'input': user_prompt})
        st.write(response['answer'])
    else:
        st.error('Please write your prompt')

```

## Conclusion

By following these steps, you’ve created a robust Document QA application using Streamlit and Groq. This application not only showcases the power of modern NLP tools but also provides a user-friendly interface for efficiently querying complex documents.

Disclaimer

The user is not responsible for any type of content produced by the model. The responses generated by the model are based on the capabilities of the underlying language model.
