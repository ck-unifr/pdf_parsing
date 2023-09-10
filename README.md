# PDF解析
![Python](https://img.shields.io/badge/Python-3.9-blue) ![PyPDF2](https://img.shields.io/badge/PyPDF2-3.0.1-blue) ![PyMuPDF](https://img.shields.io/badge/PyMuPDF-1.23.3-blue)  ![Langchain](https://img.shields.io/badge/Langchain-0.0.285-blue)  ![Rwkv](https://img.shields.io/badge/RWKV-0.8.12-blue) ![Pandas](https://img.shields.io/badge/Pandas-2.1.0-blue) ![Ninja](https://img.shields.io/badge/Ninja-1.11.1-blue)


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

并且在这个项目使用了[RWKV-Raven-7B](https://huggingface.co/BlinkDL/rwkv-4-raven)对PDF做摘要。

在项目中，有以下一个主要文件：
- src/parser.py：包含了所有 PDF 解析相关代码。
- src/llm_summarizer.py：包含了大模型摘要相关代码。
- src/main.py：包含了一些示例代码，展示了如何使用 src/parser.py 中的功能。

## 使用
具体例子请参考 src/main.py

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
json_file_path = 'home/text/sections.json'
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

**获取摘要**
```
llm_summarizer = LLMSummarizer()
summary = llm_summarizer.summarize(pdf_path)
```


## 总结
由于时间关系目前的PDF解析还存在需要优化的地方。
- 表格解析：开发中发现表格解析很有挑战。目前使用的库是[PyMuPDF](https://pymupdf.readthedocs.io/en/latest/)，还是有不少表格提错的地方，计划尝试其他多模态的框架，例如 [LayoutLM](https://huggingface.co/docs/transformers/model_doc/layoutlm)。
- 图表解析：可以尝试一些基于大模型的库可来解析图表，例如 [DePlot](https://huggingface.co/docs/transformers/main/model_doc/deplot)。
- 参考目前只是把每一条参考都输出，没有对每一天参考做结构化（时间，作者，标题），参考条目的结构化是接下去的一个工作，可以尝试使用大模型做这个结构化的工作。
- PDF结构化后，可以用大模型对结构化好的部位做问答，为了让问答更有效率我们需要将结构化的内容做切分然后存入向量库。
