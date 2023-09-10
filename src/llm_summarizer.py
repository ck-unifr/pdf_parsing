# -*- coding: utf-8 -*-
# 作者：陈凯
# 电子邮件：chenkai0210@hotmail.com
# 日期：2023-09
# 描述：这个脚本实现了基于大模型对PDF做摘要。

from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.chains import LLMChain
from langchain.llms import RWKV
import os
os.environ["RWKV_CUDA_ON"] = '1'
os.environ["RWKV_JIT_ON"] = '1'


class LLMSummarizer:
    """
    用大模型对PDF进行总结
    这里用到的大模型是rwkv raven 4
    https://huggingface.co/BlinkDL/rwkv-4-raven
    """

    def __init__(self):
        # @param {"type":"string"}
        self.strategy = "cuda fp16i8 *20 -> cuda fp16"
        self.model_path = '/data/model/rwkv_model/RWKV-4-Raven-7B-v12-Eng49%-Chn49%-Jpn1%-Other1%-20230530-ctx8192.pth'
        self.tokens_path = '/data/model/rwkv_model/20B_tokenizer.json'
        self.model = RWKV(model=self.model_path,
                          strategy=self.strategy,
                          tokens_path=self.tokens_path)

        self.task = """
        Below is an instruction that describes a task. Write a response that appropriately completes the request.
        # Instruction:
        Write a concise summary of the following:
        {text}
        # Response:
        CONCISE SUMMARY:
        """
        self.prompt = PromptTemplate(
            input_variables=["text"],
            template=self.task,
        )
        self.chain = LLMChain(llm=self.model, prompt=self.prompt)

    def summarize(self, pdf_path):
        loader = PyPDFLoader(pdf_path)
        # 获取第一页的前500的单子做摘要
        data = loader.load()[0]
        instruction = data.page_content[:500]  # @param {type:"string"}
        summary = self.chain.run(instruction)
        return summary
