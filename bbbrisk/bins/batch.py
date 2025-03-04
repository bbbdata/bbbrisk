# -*- coding: utf-8 -*-
"""
本代码用于提供一些日常业务中需要使用的分箱相关的方法
"""
from  . import merge
from  .Bins import Bins


# 自动分箱
def _autoBinOne(x,y,is_enum):
    if(is_enum==True):                                    # 如果是枚举值
        bin_set = merge.chi2Enum(x, y)                    # 使用卡方枚举分箱
    else:                                                 # 如果是连续变量
        bin_set= merge.chi2(x, y)                         # 使用卡方分箱
    return bin_set


def autoBins(x,y,enum_var=[],one_fcn=None):
    bin_sets = {}                                         # 初始化分箱统计
    for i in range(x.shape[1]):                           # 逐个变量自动分箱
        var     = x.columns[i]                            # 当前变量的名称
        cur_x   = x.iloc[:,i]                             # 变量的数据
        is_enum = var in enum_var                         # 变量是否枚举变量
        if (one_fcn==None):                               # 如果没有配置单变量分箱函数
            one_fcn = _autoBinOne                         # 使用默认分箱函数
        bin_set = one_fcn(cur_x,y,is_enum)                # 对当前变量分箱
        bin_sets[var] = bin_set                           # 记录分箱结果
    return bin_sets                                       # 返回分箱结果

# 分箱应用
def bin_stats(x,y,bin_sets):
    bin_stats = {}                                        # 初始化分箱统计
    for var,bset in bin_sets.items():                     # 逐个变量循环
        stat = Bins(bset).binStat(x[var],y)               # 初始化当前变量的分箱类
        bin_stats[var] = stat                             # 记录当前变量的分箱统计
    return bin_stats                                      # 返回分箱结果



