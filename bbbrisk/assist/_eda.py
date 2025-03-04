# -*- coding: utf-8 -*-
"""
本代码用于数据探索
"""
import pandas as pd
from .. import encode
from sklearn.decomposition import PCA

# 探索数据的线性相关的信息
def linear_info(x,y,data_trans=None,bin_sets=None):                  
    if(data_trans==None):                                      # 如果是iden
        xx = x                                                 # 则直接分析x本身
    elif(data_trans=='grp'):                                   # 如果类型是grp
        xx,_ = encode.grp.to_grp(x,bin_sets)                   # 则将数据转换为分组数据
    elif(data_trans=='woe'):                                   # 如果类型是woe
        xx,_ = encode.grp.to_grp(x,bin_sets)                   # 先将数据转换为分组数据
        xx,_ = encode.woe.to_woe(x,y)                          # 再将分组数据转换为woe数据
														       
	# 计算主成份贡献信息                                       
    clf  = PCA().fit(xx)                                       # PCA分析
    pr   = clf.explained_variance_ratio_                       # 贡献占比
    pca_info = pd.DataFrame({'pr':pr,'pr_csum':pr.cumsum()})   # 贡献占比与累计贡献占比
	
	# 计算相关系数
    corr_info = pd.concat([xx,y],axis=1).corr()                # 计算相关系数矩阵
    return pca_info,corr_info                                  # 返回主成份贡献与相关系数矩阵



