import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os

from typing import Any, List, Mapping, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM

from transformers import AutoTokenizer, AutoModel
from utils import get_config_variable

from utils import get_config_variable
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.chains import LLMChain
from langchain.llms import RWKV
import os

os.environ["RWKV_CUDA_ON"] = '1'
os.environ["RWKV_JIT_ON"] = '1'


# Sidebar contents
with st.sidebar:
    st.title('🤗🤗💬 LLM Chat App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    ''')
    add_vertical_space(5)
    st.write('Made by Kai Chen')


@st.experimental_singleton
def load_llm():
    print('load llm')
    strategy = "cuda fp16i8 *20 -> cuda fp16"
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    config_file = f'{ROOT_DIR[:-3]}config.ini'  # 配置文件的路径
    model_path = get_config_variable(config_file, 'LLM', 'rwkv_model_path')
    tokens_path = get_config_variable(
        config_file, 'LLM', 'rwkv_tokenizer_path')
    model = RWKV(model=model_path,
                 strategy=strategy,
                 tokens_path=tokens_path)
    return model


def main():
    st.header("Chat with PDF 💬")
    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')
    # st.write(pdf)
    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        # # embeddings
        store_name = pdf.name[:-4]
        st.write(f'{store_name}')
        # st.write(chunks)

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
            # st.write('Embeddings Loaded from the Disk')s
        else:
            # embeddings = OpenAIEmbeddings()
            embeddings = HuggingFaceEmbeddings(
                model_name='/data/model/all-MiniLM-L6-v2')
            # st.write(embeddings)
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)

        # embeddings = OpenAIEmbeddings()
        # VectorStore = FAISS.from_texts(chunks, embedding=embeddings)

        # Accept user questions/query
        query = st.text_input("Ask questions about your PDF file:")
        st.write(query)

        if query:
            docs = VectorStore.similarity_search(query=query, k=3)
            st.write(docs)

            # llm = OpenAI()
            # llm = CustomLLM()
            llm = load_llm()
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)


if __name__ == '__main__':
    main()
