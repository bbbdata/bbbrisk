# -*- coding: utf-8 -*-
"""
本代码用于封装各种分箱算法,供用户使用
"""

import pandas as pd
from . import _merge

# 通用方法-将分割点转为分箱格式
def _cut_to_bin(x,cut):
    if(len(cut)==0):                                                         # 如果没有切割点
        return [['-','+']]                                                   # 则返回负无穷到正无穷
    binset = [['-',cut[0]]]                                                  # 初始化第一个分箱
    binset = binset+[[cut[i],cut[i+1]] for  i in range(len(cut)-1)]          # 添加其余分箱
    binset = binset +[[cut[-1],'+']]                                         # 添加最后一个分箱
    if(any(pd.isnull(x))):                                                   # 如果有空值
        binset.append(None)                                                  # 添加空值分箱
    return binset                                                            # 返回结果
																		     
'''连续变量的分箱方法'''                                                     
# 等频分箱                                                                   
def eSample(x,bin_sample=100):                                               # 等频分箱
    cut = _merge._eSampleBin.eSampleBin(x,bin_sample)                        # 直接调用等频分箱算法获取切割点
    return  _cut_to_bin(x,cut)                                               # 将切割点转换为分箱,返回结果
																		     
# 等距分箱                                                                   
def eDist(x,bin_num=10):                                                     # 等距分箱
    cut = _merge._eDistBin.eDistBin(x,bin_num)                               # 直接调用等距分箱算法获取切割点
    return  _cut_to_bin(x,cut)                                               # 将切割点转换为分箱,返回结果
																		     
# 幂分箱                                                                   
def powerDist(a=-1,b=4):                                                     # 幂分箱
    cut = []                                                                 # 初始化切割点
    for k in range(a,b):                                                     # 逐幂次循环
        for i in range(1,10):                                                # 将当前幂次分为10个箱
            v = round(i*(10**k),-k) if k<0 else i*(10**k)                    # 当前幂次的切割点
            cut.append(v)                                                    # 添加当前幂次的切割点
    binset = [['-',cut[0]]]                                                  # 将<=第一个切割点作为一个分箱
    binset = binset+[[cut[i],cut[i+1]] for  i in range(len(cut)-1)]          # 添加其余分箱
    binset = binset +[[cut[-1],'+']]                                         # 将>最后一个切割点作为一个分箱
    return binset                                                            # 返回结果

# 卡方分箱                                                                   
def chi2(x,y,bin_num = 5,init_bin_num=10):                                   # 卡方分箱
    cut = _merge._chi2Bin.chi2Merge(x,y,bin_num = 5,init_bin_num=10)         # 直接调用卡方分箱算法获取切割点
    return  _cut_to_bin(x,cut)                                               # 将切割点转换为分箱,返回结果
																			 
# ks分箱                                                                     
def ks(x,y,bin_num=5,min_sample=None):                                       # ks分箱
    cut =  _merge._ksBin.ksMerge(x,y,bin_num,min_sample)                     # 直接调用ks分箱算法获取切割点
    return  _cut_to_bin(x,cut)                                               # 将切割点转换为分箱,返回结果
													 
# 决策树分箱                                                                     
def tree(x,y,max_depth=3,min_sample=None):                                   # 决策树分箱
    cut =  _merge._treeBin.treeMerge(x,y,max_depth,min_sample)               # 直接调用决策树分箱算法获取切割点
    return  _cut_to_bin(x,cut)                                               # 将切割点转换为分箱,返回结果
	

																		
'''枚举变量的分箱方法'''                                                      
# 卡方分箱(枚举)                                                              
def chi2Enum(x,y,bin_num = 5,init_bin_num=10):                               # 卡方分箱(枚举)
    return  _merge._chi2Bin.chi2MergeEnum(x,y,bin_num = 5,init_bin_num=10)   # 直接调用卡方分箱算法获取分箱结果
																			 
# ks分箱(枚举)                                                               
def ksEnum(x,y,bin_num=5,min_sample=None):                                   # ks分箱(枚举)
    return _merge._ksBin.ksMergeEnum(x,y,bin_num,min_sample)                 # 直接调用ks分箱算法获取分箱结果

# 决策树分箱                                                                     
def treeEnum(x,y,max_depth=3,min_sample=None):                               # 决策树分箱
    return _merge._treeBin.treeMergeEnum(x,y,max_depth,min_sample)           # 直接调用决策树分箱算法获取分箱结果
																			 
# 所有枚举值                                                                 
def allEnum(x):                                                              # 所有枚举值 
    return _merge._allEnumBin.allEnum(x)                                     # 直接调用"所有枚举值"分箱算法获取分箱结果

