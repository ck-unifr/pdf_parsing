# PDF解析
![Python](https://img.shields.io/badge/Python-3.9-blue) ![PyPDF2](https://img.shields.io/badge/PyPDF2-3.0.1-blue) ![PyMuPDF](https://img.shields.io/badge/PyPDF2-1.23.3-blue) ![Pandas](https://img.shields.io/badge/Pandas-2.1.0-blue) ![Langchain](https://img.shields.io/badge/Langchain-0.0.285-blue) ![Ninja](https://img.shields.io/badge/Ninja-1.11.1-blue) ![Rwkv](https://img.shields.io/badge/RWKV-0.8.12-blue)


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

并且在这个项目使用了RWKV对PDF做摘要。

在项目中，有两个主要文件：
- src/parser.py：包含了所有 PDF 解析相关代码。
- src/llm_summarizer.py：包含了大模型摘要相关代码。
- src/main.py：包含了一些示例代码，展示了如何使用 src/parser.py 中的功能。

## 总结
时间关系目前的PDF解析还有需要优化的地方。
- 表格解析：开发中发现表格解析很有挑战。目前使用的库是[PyMuPDF](https://pymupdf.readthedocs.io/en/latest/)，还是有不少表格提错的地方，计划尝试其他多模态的框架，例如 [LayoutLM](https://huggingface.co/docs/transformers/model_doc/layoutlm)。
- 图表解析：可以尝试一些基于大模型的库可来解析图表，例如 [DePlot](https://huggingface.co/docs/transformers/main/model_doc/deplot)。
- 参考目前只是把每一条参考都输出，没有对每一天参考做结构化（时间，作者，标题），参考条目的结构化是接下去的一个工作，可以尝试使用大模型做这个结构化的工作。
- PDF结构化后，可以用大模型对结构化好的部位做问答，为了让问答更有效率我们需要将结构化的内容做切分然后存入向量库。