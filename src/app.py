import os
from typing import Any, List, Mapping, Optional
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
import pickle
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.chains import LLMChain
from langchain.llms import RWKV
from transformers import AutoTokenizer, AutoModel, AutoConfig
from utils import get_config_variable


os.environ["RWKV_CUDA_ON"] = '1'
os.environ["RWKV_JIT_ON"] = '1'


class GLM(LLM):
    max_token: int = 2048
    temperature: float = 0.8
    top_p = 0.9
    tokenizer: object = None
    model: object = None
    history_len: int = 1024

    def __init__(self):
        super().__init__()

    @property
    def _llm_type(self) -> str:
        return "GLM"

    def load_model(self, llm_device="gpu", model_name_or_path=None):
        model_config = AutoConfig.from_pretrained(
            model_name_or_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name_or_path,
                                               config=model_config,
                                               trust_remote_code=True).half().cuda()

    def _call(self, prompt: str, history: List[str] = [], stop: Optional[List[str]] = None):
        response, _ = self.model.chat(
            self.tokenizer, prompt,
            history=history[-self.history_len:] if self.history_len > 0 else [],
            max_length=self.max_token, temperature=self.temperature,
            top_p=self.top_p)
        return response


# Sidebar contents
with st.sidebar:
    st.title('🤗🤗💬 LLM Chat App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [ChatGLM2-6B](https://github.com/THUDM/ChatGLM2-6B)
    ''')
    add_vertical_space(5)
    st.write('Made by Kai Chen')


@st.experimental_singleton
def load_llm_chatglm():
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    config_file = f'{ROOT_DIR[:-3]}config.ini'  # 配置文件的路径
    model_path = get_config_variable(config_file, 'LLM', 'chatglm2_6b_path')
    llm = GLM()
    llm.load_model(model_name_or_path=model_path)
    return llm


@st.experimental_singleton
def load_llm_rwkv():
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
            # llm = load_llm_rwkv()
            llm = load_llm_chatglm()
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)


if __name__ == '__main__':
    main()
