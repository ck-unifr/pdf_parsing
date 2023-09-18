# PDF解析
![Python](https://img.shields.io/badge/Python-3.9-blue) ![PyPDF2](https://img.shields.io/badge/PyPDF2-3.0.1-blue) ![PyMuPDF](https://img.shields.io/badge/PyMuPDF-1.23.3-blue)  ![Langchain](https://img.shields.io/badge/Langchain-0.0.285-blue)  ![Rwkv](https://img.shields.io/badge/RWKV-0.8.12-blue) ![ChatGLM2](https://img.shields.io/badge/ChatGLM-2-blue) ![Pandas](https://img.shields.io/badge/Pandas-2.1.0-blue) ![Ninja](https://img.shields.io/badge/Ninja-1.11.1-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-1.26.0-blue)

## 介绍

实现对PDF解析，将给定的PDF结构化成以下几个部分。
- 文字
  - 总标题，章节标题和章节对应的文字内容
- 图片
  - 图片和图片标题
- 表格
  - 表格和表格标题
- 参考
  - 参考

在这个项目中还有两个部分用到了大模型
- 使用了[RWKV-Raven-7B](https://huggingface.co/BlinkDL/rwkv-4-raven)对PDF做摘要。
- 是用了[ChatGLM2-6B](https://github.com/THUDM/ChatGLM2-6B)对参考文献做信息抽取。
将参考文献结构化成字典的格式，字典包含了”作者“，”标题“，”年份“。

在这个项目中还有实现了一个对PDF问答的例子。

以下是这个项目的几个主要文件：
- ```src/pdf_parser.py```：包含了所有 PDF 解析相关代码。
- ```src/llm_summarizer.py```：包含了大模型摘要相关代码。
- ```src/llm_extractor.py```：包含了大模型对参考文献做信息抽取相关代码。
- ```src/main.py```：包含了一些示例代码，展示了如何使用 ```src/pdf_parser.py``` 中的功能。
- ```src/utils.py```: 包含了一些工具函数。
- ```src/app.py```: 包含了一个用```streamlit```和```langchain```做PDF问答的例子。
- ```config.ini```：包含了大模型文件路径和相关的tokenizer文件路径。


## 使用

关于PDF解析的具体例子请参考 ```src/main.py```。
关于PDF问答请参考```src/app.py```。


**初始化**

首先初始化一个类并讲需要解析的PDF文件路径传入到该类。

```
from parser import PDFParser

pdf_path = '/home/data/gpt-4.pdf'
parser = PDFParser(pdf_path)
```

**获取文字：标题，章节名称和对应的文字内容**

```
import json

parser.extract_text()
# 指定保存的文件路径
json_file_path = '/home/text/sections.json'
with open(json_file_path, 'w') as json_file:
    json.dump(parser.text.section, json_file)
```

**获取图片：图片和对应的标题**

```
parser.extract_images()

images = parser.images
for image in images:
    # 将图像保存为文件
    image_filename = f"/home/image/image_{image.page_num}_{image.title[:10]}.png"
    with open(image_filename, "wb") as image_file:
        logging.info(image.title)
        logging.info(image.page_num)
        image_file.write(image.image_data)
```

**获取表格：表格和对应的标题**

```
parser.extract_tables()
for i, table in enumerate(parser.tables):
    csv_filename = "/home/table/table_i_{table.page_num}_{table.title[:10]}.csv"
    table.table_data.to_csv(csv_filename)
```

**获取参考**

```
parser.extract_references()
with open('/home/reference/references.txt', 'w') as fp:
    for ref in parser.references:
        fp.write("%s\n" % ref.ref)
```

**获取参考的信息：作者，标题，年份**

```
from parser import PDFParser
from llm_extractor import LLMExtractor
from tqdm import tqdm

parser.extract_references()

llm_extractor = LLMExtractor()
for i, ref in enumerate(tqdm(parser.references)):
    json_ref = llm_extractor.extract_reference(ref)
    if json_ref and len(json_ref) > 0:
        with open('/home/reference/{i}.json', 'w') as outfile:
            json.dump(json_ref, outfile)
```

**获取摘要**

```
from llm_summarizer import LLMSummarizer

llm_summarizer = LLMSummarizer()
summary = llm_summarizer.summarize(pdf_path)
```

**运行PDF问答**
```
streamlit run app.py --server.fileWatcherType none
```


这个项目用到的是大模型是[RWKV-Raven-7B](https://huggingface.co/BlinkDL/rwkv-4-raven)，
[ChatGLM2-6B](https://github.com/THUDM/ChatGLM2-6B)
需要在config.ini文件中设置相关的模型文件路径和tokenizer文件路径。
以下是config.ini的文件内容

```
[LLM]
rwkv_model_path=/data/model/rwkv_model/RWKV-4-Raven-7B-v12-Eng49%%-Chn49%%-Jpn1%%-Other1%%-20230530-ctx8192.pth
rwkv_tokenizer_path=/data/model/rwkv_model/20B_tokenizer.json
chatglm2_6b_path=/data/model/chatglm2-6b
```

## 总结

由于时间关系目前的PDF解析还存在需要优化的地方。
- 表格解析：开发中发现表格解析很有挑战。
目前使用的库是[PyMuPDF](https://pymupdf.readthedocs.io/en/latest/)，还是有不少表格提错的地方，计划尝试其他多模态的框架，
例如 [LayoutLM](https://huggingface.co/docs/transformers/model_doc/layoutlm) [table-transformer](https://github.com/microsoft/table-transformer) [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/ppstructure/table/README.md)。
- 图表解析：可以尝试一些基于大模型的库可来解析图表，例如 [DePlot](https://huggingface.co/docs/transformers/main/model_doc/deplot)。
- 时间关系这里用了两个不同的大模型：```RWKV-Raven-7B```和```ChatGLM2-6B``` 分别做摘要和信息抽取，可以考虑只用一个大模型。
- PDF结构化后，可以用大模型对结构化好的部位做问答，为了让问答更有效率我们需要将结构化的内容做切分然后存入向量库。