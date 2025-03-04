# -*- coding: utf-8 -*-
"""
本代码用于实现:卡方分箱
本代码由《老饼讲解-机器学习》www.bbbdata.com编写
"""
import numpy as np
import pandas as pd
from . import _eSampleBin

'''计算卡方值'''                                                                  
# chi2_value,p,free_n,ex = scipy.stats.chi2_contingency(pair)                     # 可以使用scipy直接计算卡方值,这里自行实现
def _cal_chi2(pair):                                                               
    pair = np.array(pair)                                                         # 转为numpy
    pair[pair==0] = 1                                                             # 转换为1,避免分母为0
    class_rate    = pair.sum(axis=1)/pair.sum().sum()                             # 两类样本的占比
    col_sum       = pair.sum(axis=0)                                              # 各组别的样本个数
    ex            = np.dot(np.array([class_rate]).T,np.array([col_sum]))          # 计算期望值
    chi2          = (((pair - ex)**2/ex)).sum()                                   # 计算卡方值
    return chi2                                                                   

																		          																	          
'''卡方分箱主函数(连续变量)'''                                                    
def chi2Merge(x,y,bin_num = 5,init_bin_num=10):                                   
																		          
    # -------初始化------------------                                             
    # 用等频分箱初始化分箱                                                        
    bin_sample = int(len(x)/init_bin_num)                                         # 等频分箱的样本个数
    bin_cut    =  _eSampleBin.eSampleBin(x,bin_sample = bin_sample)                # 初始化分箱
    bin_cut.append(max(x))                                                        # 添加最大值作为最后的切割点   
																			      
    # 初始化每个分箱的个数统计                                                    
    grp = np.arange(len(bin_cut))                                                 # 组号
    df  = pd.DataFrame({'grp':grp,'cut':bin_cut})                                 # 初始化分箱统计 
    for i in range(df.shape[0]):                                                  # 逐组统计
        idx = x<=bin_cut[i] if i == 0  else (x>bin_cut[i-1])&(x<=bin_cut[i])      # 符合本组的样本索引
        df.loc[i,'good_cn'] = ((idx)&(y==0)).sum()                                # 统计好样本个数
        df.loc[i,'bad_cn']  = ((idx)&(y==1)).sum()                                # 统计坏样本个数
        
    # ----根据卡方值合并分箱，直到达到目标分箱数---------
    c_name = ['good_cn','bad_cn']                                                 # 各个类别个数统计时用的名称
    while(df.shape[0]>bin_num):                                                   # 当箱数大于目标箱数时,合并卡方值最小的两个分箱
        chi2_list = [_cal_chi2(df[c_name][i:i+2]) for i in range(df.shape[0]-1)]  # 计算卡方值
        min_idx   = np.argmin(chi2_list)                                          # 最小卡方值的索引
        df.loc[min_idx+1,c_name] += df.loc[min_idx,c_name]                        # 将最小卡方值合并到下一个分箱
        df.drop(min_idx, inplace = True)                                          # 删掉最小卡方值分箱
        df = df.reset_index(drop=True)                                            # 重置索引
    bin_cut = np.array(df['cut'][:-1])                                            # 获取切割点
    min_gap = min(bin_cut[1:]-bin_cut[:-1]) if len(bin_cut)>1 else 0			  # 分箱之间的最小间隔
    if(abs(min_gap)>0.0001):                                                      # 如果间隔足够
        bin_cut = bin_cut.round(4)                                                # 进行截断
    return list(bin_cut)                                                          # 返回分箱切割点与p值


'''卡方分箱主函数(枚举变量)'''
def chi2MergeEnum(x,y,bin_num = 5,init_bin_num=10):
    
    # 转为badrate,再按连续变量进行分箱
    data = pd.DataFrame({'x':x,'_is_bad':y })                                     # 将x,y拼接
    df  = data.groupby(x)['_is_bad'].agg([('bad_cn','sum'),('cn','count')])       # 按组别统计坏样本个数与总样本个数
    df = pd.DataFrame({'grp':df.index,'bad_rate':df['bad_cn']/df['cn']})          # 计算bad_rate
    bad_rate_dict = dict(zip(df['grp'], df['bad_rate']))                          # 转为字典:组别与bad_rate
    x_br   = data['x'].map(bad_rate_dict)                                         # 将x转换为bad_rate数据
    bin_cut = chi2Merge(x_br,y,bin_num,init_bin_num)                              # 使用卡方分箱(连续变量)进行分箱
																		          
    # 按照分割结果,找出每个切割点对应的枚举值                                     
    bin_set = [tuple(df.loc[df['bad_rate']<=bin_cut[0],'grp'])]                   # 第一个分箱的枚举值
    for i in range(len(bin_cut)-1):                                               # 逐个分箱拼接
        idx = ( df['bad_rate']>bin_cut[i])&( df['bad_rate']<=bin_cut[i+1])        # 第i+1个分箱的     
        bin_set.append(tuple(df.loc[idx,'grp']))                                  # 第i+1个分箱的枚举值 
    bin_set.append(tuple(df.loc[ df['bad_rate']>bin_cut[-1],'grp']))              # 拼接最后一个分箱
    return bin_set                                                                # 返回分箱结果

