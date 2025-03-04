# -*- coding: utf-8 -*-
"""
本代码用于加载数据集
"""

import os 
import pandas as pd

# 加载原始小贷数据
def load_bloan():
    module_path = os.path.dirname(__file__)                        # 读取当前文件名的位置
    path = os.path.join(module_path,'bloan.csv')                   # 拼接出CSV文件的完整路径
    data = pd.read_csv(path)                                       # 读取数据
    return data                                                    # 返回数据


# 加载变量分箱后的小贷数据
def load_bloan_grp():
    module_path = os.path.dirname(__file__)                        # 读取当前文件名的位置
    path = os.path.join(module_path,'bloan_grp.csv')               # 拼接出CSV文件的完整路径
    data = pd.read_csv(path)                                       # 读取数据
    return data                                                    # 返回数据
    