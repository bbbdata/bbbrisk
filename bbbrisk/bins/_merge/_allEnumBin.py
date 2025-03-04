# -*- coding: utf-8 -*-
"""
本代码用于实现:枚举分箱
本代码由《老饼讲解-机器学习》www.bbbdata.com编写
"""

'''以枚举变量的所有枚举值作为分箱'''
def allEnum(x):
    y = None                   # 初始化y
    s = set(x)                 # 对x去重
    if None in s:              # 如果含有None值
        s.discard(None)        # 先删除None值
        y = sorted(list(s))    # 排序 
        y.append(None)         # 把None值拼接上
    else:                      # 如果没有None值
        y = sorted(list(s))    # 直接排序 
    return y                   # 返回结果