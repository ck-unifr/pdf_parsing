# -*- coding: utf-8 -*-
# 作者：陈凯
# 电子邮件：chenkai0210@hotmail.com
# 日期：2023-09
# 描述：一些工具方法

import configparser


def get_config_variable(config_file: str, section: str, variable_name: str):
    """
    从配置文件中获取变量值

    Args:
        config_file (str): 配置文件的路径
        section (str): 配置文件中的节名
        variable_name (str): 要获取的变量名

    Returns:
        str: 变量的值，如果找不到则返回 None
    """
    config = configparser.ConfigParser()
    config.read(config_file)

    if config.has_section(section):
        if config.has_option(section, variable_name):
            return config.get(section, variable_name)

    return None
