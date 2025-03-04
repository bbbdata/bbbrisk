# -*- coding: utf-8 -*-
"""
本代码用于实现:等频分箱
本代码由《老饼讲解-机器学习》www.bbbdata.com编写
"""
import numpy as np

'''等频分箱'''
def eSampleBin(x,bin_sample=100):   
    x = np.array(x)
    x = x[~np.isnan(x)]                             # 屏蔽nan值
    x.sort()                                        # 排序
    cut = []                                        # 初始化切割
    start_idx = -1                                  # 初始化起始位置
    xlen = len(x)                                   # 样本个数(已经去除了空值)
    for i in range(xlen):                           # 逐个获取切割点(最多不超过xlen个切割点)
        start_idx = start_idx+bin_sample            # 从起始点向前读min_sample个位置
        if (start_idx>=xlen):                       # 如果已经超出数据个数
           break                                    # 则不需再切割,直接退出
        cut_val = x[start_idx]                      # 获取当前的切割点
        cut.append(cut_val)                         # 记录当前切割点
        start_idx = np.where(x==cut_val)[0].max()   # 将起始点推向当前切割值的最后一个位置
    return cut