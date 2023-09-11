# -*- coding: utf-8 -*-
# 作者：陈凯
# 电子邮件：chenkai0210@hotmail.com
# 日期：2023-09
# 描述：这个脚本实现了基于大模型对输入的文字做信息抽取。

import os
import json
from transformers import AutoTokenizer, AutoModel
from utils import get_config_variable


class LLMExtractor:
    """
    用大模型对文字做信息抽取
    这里用到的大模型是chatglm2 6b
    https://github.com/THUDM/ChatGLM2-6B/tree/main
    """

    def __init__(self):
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        config_file = f'{ROOT_DIR[:-3]}config.ini'  # 配置文件的路径
        self.model_path = get_config_variable(
            config_file, 'LLM', 'chatglm2_6b_path')
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            self.model_path, trust_remote_code=True, device='cuda').half().cuda()

    @staticmethod
    def get_prompt(content):
        data = {
            "author": "Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D. Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell",
            "title": "Language models are few-shot learners",
            "year": "2020"
        }

        # 使用ensure_ascii=False以支持非ASCII字符
        json_string = json.dumps(data, ensure_ascii=False)

        prompt = f"""
        从输入的文字中，提取"信息"(keyword,content)，包括:"author"、"title"、"year"的实体，输出json格式内容。
        请只输出json，不需要输出其他内容。
        
        例如：
        输入
        [1] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D. Kaplan, Prafulla Dhariwal,Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models arefew-shot learners. Advances in Neural Information Processing Systems, 33:1877–1901, 2020.

        输出
        {json_string}

        输入
        {content}

        输出
        以下是提取的实体及其JSON格式内容：
        """

        return prompt

    def extract_reference(self, ref: str) -> dict:
        prompt = LLMExtractor.get_prompt(ref)
        if self.model:
            response, history = self.model.chat(
                self.tokenizer, prompt, history=[])
            try:
                json_object = json.loads(response)
                return json_object
            except:
                # TODO: 错误分析
                pass
        return None
