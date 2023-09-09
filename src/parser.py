# -*- coding: utf-8 -*-
# 作者：陈凯
# 电子邮件：chenkai0210@hotmail.com
# 日期：2023-09
# 描述：这个脚本的目的是解析PDF。将PDF结构化成，文字，图片，表格和参考。

from typing import List
import PyPDF2  # PDF相关操作
import fitz   # PyMuPDF PDF相关操作
import pandas as pd  # 用于结构化表格
# import tabula  # 用于提取 PDF 中的表格数据


class Text:
    def __init__(self, title: str = None, section: dict = {}):
        """
        文本类，用于封装提取的文本内容

        参数:
        - title: str，文本标题
        - section: dict, key:章节名称 value:章节文字内容
        """
        self.title = title
        self.section = section


class PDFImage:
    def __init__(self, title: str, image_data: object, page_num: int):
        """
        图片类，用于封装提取的图片信息

        参数:
        - title: str，图片标题
        - image_data: 图片数据的表示形式，可以是字节流、文件路径等
        - page_num: int: 图片所在页数
        """
        self.title = title
        self.image_data = image_data
        self.page_num = page_num


class Table:
    def __init__(self, title: str, table_data: pd.DataFrame, page_num: int):
        """
        表格类，用于封装提取的表格信息

        参数:
        - title: str，表格标题
        - table_data: 表格数据，可以是 Pandas DataFrame 等形式
        - page_num: int: 表格所在页数
        """
        self.title = title
        self.table_data = table_data
        self.page_num = page_num


class Reference:
    def __init__(self, ref: str):
        """
        参考文献类，用于封装提取的参考文献信息

        参数:
        - ref: str，参考文献
        """
        self.ref = ref


class PDFOutliner:
    """
    获取给定PDF的所有章节的标题
    该类对下面的代码做了一些修改
    https://github.com/beaverden/pdftoc/tree/main
    """

    def __init__(self):
        self.titles = []  # 每一个章节的标题

    def get_tree_pages(self, root, info, depth=0, titles=[]):
        """
            Recursively iterate the outline tree
            Find the pages pointed by the outline item
            and get the assigned physical order id
            Decrement with padding if necessary
        """
        if isinstance(root, dict):
            page = root['/Page'].get_object()
            t = root['/Title']
            title = t
            if isinstance(t, PyPDF2.generic.ByteStringObject):
                title = t.original_bytes.decode('utf8')
            title = title.strip()
            title = title.replace('\n', '')
            title = title.replace('\r', '')
            page_num = info['all_pages'].get(id(page), 0)
            if page_num == 0:
                # TODO: logging
                print('Not found page number for /Page!', page)
            elif page_num < info['padding']:
                page_num = 0
            else:
                page_num -= info['padding']
            str_val = '%-5d' % page_num
            str_val += '\t' * depth
            str_val += title + '\t' + '%3d' % page_num
            self.titles.append(title)
            return
        for elem in root:
            self.get_tree_pages(elem, info, depth+1)

    def recursive_numbering(self, obj, info):
        """
            Recursively iterate through all the pages in order and 
            assign them a physical order number
        """
        if obj['/Type'] == '/Page':
            obj_id = id(obj)
            if obj_id not in info['all_pages']:
                info['all_pages'][obj_id] = info['current_page_id']
            info['current_page_id'] += 1
            return
        elif obj['/Type'] == '/Pages':
            for page in obj['/Kids']:
                self.recursive_numbering(page.get_object(), info)

    def create_text_outline(self, pdf_path, page_number_padding):
        # print('Running the script for [%s] with padding [%d]' % (pdf_path, page_number_padding))
        # creating an object
        titles = []
        with open(pdf_path, 'rb') as file:
            fileReader = PyPDF2.PdfReader(file)

            info = {
                'all_pages': {},
                'current_page_id': 1,
                'padding': page_number_padding
            }

            pages = fileReader.trailer['/Root']['/Pages'].get_object()
            self.recursive_numbering(pages, info)
            # for page_num, page in enumerate(pages['/Kids']):
            #    page_obj = page.getObject()
            #    all_pages[id(page_obj)] = page_num + 1
            self.get_tree_pages(fileReader.outline, info, 0, titles)
        return


class PDFParser:
    def __init__(self, pdf_path: str):
        """
        PDF 解析器类，用于提取 PDF 中的文本、图片、表格和参考文献信息
        参数:
        - pdf_path: str，PDF 文件的路径
        """
        self.pdf_path = pdf_path
        self.doc = fitz.open(self.pdf_path)  # fitz.Document
        self.text = Text()   # text: Text, 文字内容
        self.images = []     # list, 所有图片（PDFImage）
        self.tables = []     # list, 所有表格（Table）
        self.references = []  # list, 所有参考（Reference）

    def extract_title(self):
        """
        获取pdf标题
        """
        doc = self.doc
        first_page = doc.load_page(0)  # 获取第一页
        # 提取第一页的文本内容
        text = first_page.get_text()
        # 按行拆分文本内容
        lines = text.split('\n')
        # 获取第一行文本
        first_line = lines[0].strip()
        self.text.title = first_line
        return

    def extract_sections_content(self,
                                 doc: fitz.Document,
                                 section_titles: List[str]):
        """
        根据章节名称列表提取PDF中各章节的文字内容。
        参数：
        - pdf_file: 包含章节的PDF文件路径。
        - section_titles: 包含所有章节名称的列表。

        返回值：
        - 一个字典，键是章节名称，值是该章节的文字内容。
        """
        sections_content = {}  # 存储章节名称和内容的字典
        filtered_section_titles = [PDFParser.remove_leading_digits(
            title).strip() for title in section_titles]
        for i, section_title in enumerate(filtered_section_titles):
            section_found = False
            section_content = ""

            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()

                for line in page_text.split('\n'):
                    if section_title.lower() in line.lower():
                        section_found = True
                    elif section_found:
                        # 如果找到了目标标题，开始获取章节内容
                        section_content += line + "\n"

            if section_found:
                sections_content[section_titles[i]] = section_content

        return sections_content

    @staticmethod
    def remove_leading_digits(text: str):
        """
        删除输入文字开头的数字。
        """
        # 用isdigit()方法检查文字的第一个字符是否是数字
        while text and text[0].isdigit():
            text = text[1:]  # 删除第一个字符
        return text

    def extract_text(self):
        """
        提取PDF中的文本内容
        """
        # 获取标题
        self.extract_title()
        # 获取章节名称
        outliner = PDFOutliner()
        outliner.create_text_outline(self.pdf_path, 0)
        # 获取对应章节下的文字内容
        self.text.section = self.extract_sections_content(
            self.doc, outliner.titles)
        return

    def extract_images(self, fig_caption_start: str = ' Figure'):
        """
        提取 PDF 中的图片信息: 图片和图片的标题
        """
        doc = self.doc

        for page_num in range(len(doc)):
            page = doc[page_num]
            # 提取页面文本块
            blocks = page.get_text('blocks')
            # 通过计算文本块与图片的距离来匹配图片和对应的标题，
            # 文本块有特定的开始词且距离离图片最近的文本块的文字为当前图片的标题
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = doc.extract_image(xref)
                x0, y0, x1, y2 = page.get_image_rects(xref)[0]
                related_text = "untitled"
                min_dist = float('inf')
                for block in blocks:
                    block_x0, block_y0, block_x1, block_y1, block_text = block[:5]
                    if block_text.strip().startswith(fig_caption_start):
                        # 计算欧式距离
                        dist = (x0 - block_x0)**2 + (y0 - block_y0)**2
                        if dist < min_dist:
                            min_dist = dist
                            related_text = block_text.strip()

                image_data = base_image["image"]
                image = PDFImage(related_text, image_data, page_num)
                self.images.append(image)

    def extract_tables(self):
        """
        提取 PDF 中的表格信息
        """
        doc = self.doc
        for num in range(len(doc)):
            page = doc[num]
            # 提取页面文本块
            blocks = page.get_text('blocks')
            # 提取表格
            tables = page.find_tables()
            # 通过计算文本块与表格的距离来匹配图片和对应的标题，
            # 文本块有特定的开始词且距离离表格最近的文本块的文字为当前图片的标题
            for table in tables:
                x0, y0, x1, y2 = table.bbox
                df = table.to_pandas()
                related_text = "untitled"
                min_dist = float('inf')
                for block in blocks:
                    block_x0, block_y0, block_x1, block_y1, block_text = block[:5]
                    if block_text.strip().startswith('Table'):
                        # 计算欧式距离
                        dist = (x0 - block_x0)**2 + (y0 - block_y0)**2
                        if dist < min_dist:
                            min_dist = dist
                            related_text = block_text.strip()
                self.tables.append(Table(title=related_text,
                                         table_data=df,
                                         page_num=num))

    def extract_references(self):
        """
        提取 PDF 中的参考文献信息
        """
        doc = self.doc
        page_num = len(doc)
        ref_list = []
        for num, page in enumerate(doc):
            content = page.get_text('blocks')
            for pc in content:
                txt_blocks = list(pc[4:-2])
                txt = ''.join(txt_blocks)
                if 'References' in txt or 'REFERENCES' in txt or 'referenCes' in txt:
                    ref_num = [i for i in range(num, page_num)]
                    for rpn in ref_num:
                        ref_page = doc[rpn]
                        ref_content = ref_page.get_text('blocks')
                        for refc in ref_content:
                            txt_blocks = list(refc[4:-2])
                            ref_list.extend(txt_blocks)
        index = 0
        for i, ref in enumerate(ref_list):
            if 'References' in ref or 'REFERENCES' in ref or 'referenCes' in ref:
                index = i
                break
        if index + 1 < len(ref_list):
            index += 1
        self.references = [Reference(ref.replace('\n', ''))
                           for ref in ref_list[index:] if len(ref) > 10]
