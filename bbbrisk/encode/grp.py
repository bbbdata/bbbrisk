
# -*- coding: utf-8 -*-
"""
本代码用于分组转换的处理
"""

import pandas as pd
from  .. import bins


# 将分箱转换为组号
def to_grp(x,bin_sets):
    bin_desc_dict = {}                                    # 初始化箱号描述字典
    x_grp    = pd.DataFrame(columns=x.columns)            # 初始化转换后的箱号数据
    for i in range(x.shape[1]):                           # 逐个变量自动分箱
        var   = x.columns[i]                              # 当前变量的名称
        cur_x = x.iloc[:,i]                               # 变量的数据
        cur_bin_set = bin_sets[var]                       # 当前的分箱配置
        b = bins.Bins(cur_bin_set)                        # 初始化分箱类
        x_grp[var]    = b._apply(cur_x)                   # 将x转换为箱号
        bin_desc_dict[var] = b.bin_desc_dict              # 记录箱号描述字典
    return x_grp,bin_desc_dict                            # 返回转换结果