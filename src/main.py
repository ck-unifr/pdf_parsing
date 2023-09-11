# -*- coding: utf-8 -*-
# 作者：陈凯
# 电子邮件：chenkai0210@hotmail.com
# 日期：2023-09
# 描述：这个脚本演示如何使用PDFParser

import os
import logging
import json
from pdf_parser import PDFParser
from llm_summarizer import LLMSummarizer
from llm_extractor import LLMExtractor
from tqdm import tqdm

if __name__ == '__main__':
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    logging.basicConfig(filename=f'{ROOT_DIR[:-3]}basic.log',
                        encoding='utf-8',
                        level=logging.INFO,
                        filemode='w',
                        format='%(process)d-%(levelname)s-%(message)s')
    logging.getLogger().addHandler(logging.StreamHandler())

    logging.info('** pdf parsing **')
    # 1 创建 PDFParser 对象
    # https://cdn.openai.com/papers/gpt-4.pdf
    pdf_path = f'{ROOT_DIR[:-3]}/data/gpt-4.pdf'
    parser = PDFParser(pdf_path)

    # 2 文字: 标题，章节目录，章节对应的文字内容
    logging.info('== extract text ==')
    parser.extract_text()
    logging.info('-- title --')
    logging.info(parser.text.title)
    logging.info('-- section --')
    for title, section in parser.text.section.items():
        logging.info(title)
    # 指定保存的文件路径
    json_file_path = f"{ROOT_DIR[:-3]}/temp/json/sections.json"
    # 使用 json.dump() 将字典保存为 JSON 文件
    with open(json_file_path, 'w') as json_file:
        json.dump(parser.text.section, json_file)

    # 3 图片
    logging.info('== extract image ==')
    parser.extract_images()
    images = parser.images
    for image in images:
        # 将图像保存为文件
        image_filename = f"{ROOT_DIR[:-3]}/temp/image/image_{image.page_num}_{image.title[:10]}.png"
        with open(image_filename, "wb") as image_file:
            logging.info(image.title)
            logging.info(image.page_num)
            image_file.write(image.image_data)

    # 4 表格：表格和对应的标题
    logging.info('== extract table ==')
    parser.extract_tables()
    for i, table in enumerate(parser.tables):
        logging.info(table.title)
        csv_filename = f"{ROOT_DIR[:-3]}/temp/table/table_i_{table.page_num}_{table.title[:10]}.csv"
        table.table_data.to_csv(csv_filename)

    # 5 参考文献
    logging.info('== extract references ==')
    parser.extract_references()
    # for ref in parser.references:
    #     print(ref.ref)
    logging.info(len(parser.references))
    with open(f'{ROOT_DIR[:-3]}/temp/reference/references.txt', 'w') as fp:
        for ref in parser.references:
            # write each item on a new line
            fp.write("%s\n" % ref.ref)

    # 6 总结
    logging.info('== summarizing (LLM) ==')
    llm_summarizer = LLMSummarizer()
    parser.text.summary = llm_summarizer.summarize(pdf_path)
    logging.info(parser.text.summary)

    # 7 用大模型将参考文献结构化成作者，标题，日期
    logging.info('== extract info (author, title, year) from references ==')
    llm_extractor = LLMExtractor()
    extracted_ref = 0
    for i, ref in enumerate(tqdm(parser.references)):
        json_ref = llm_extractor.extract_reference(ref)
        if json_ref and len(json_ref) > 0:
            with open(f'{ROOT_DIR[:-3]}/temp/reference/{i}.json', 'w') as outfile:
                json.dump(json_ref, outfile)
                extracted_ref += 1
                if extracted_ref > 5:
                    break
