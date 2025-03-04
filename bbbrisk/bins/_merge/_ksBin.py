# -*- coding: utf-8 -*-
"""
本代码用于实现:最大KS分箱
本代码由《老饼讲解-机器学习》www.bbbdata.com编写
"""
from sklearn.metrics import roc_curve
import numpy as np
import pandas as pd

# 获取ks切割点
def getKsCutPoint(x,y):                                                            
    fpr, tpr, thresholds= roc_curve(y, x)                                          # 计算fpr,tpr
    ks_idx = np.argmax(abs(fpr-tpr))                                               # 计算最大ks所在位置
    # 由于roc_curve给出的切割点是>=作为右箱,                                       
    # 而我们需要的是>作为右箱,所以切割点应向下再取一位,也即索引向上取一位          
																				   
    return thresholds[ks_idx+1]                                                    # 返回切割点
																		           
# 检查切割点是否有效                                                               
def checkCutValid(x,y,cutPoint,woe_asc,min_sample):                                
    left_y           = y[x<=cutPoint]                                              # 左箱的y        
    right_y          = y[x>cutPoint]                                               # 右箱的y  
    check_sample_num = min(len(left_y),len(right_y))>=min_sample                   # 检查左右箱样本是否足够
    left_rate        = sum(left_y)/max((len(left_y)-sum(left_y)),1)                # 左箱好坏比例
    right_rate       = sum(right_y)/max((len(right_y)-sum(right_y)),1)             # 右箱好坏比例
    cur_woe_asc      = left_rate<right_rate                                        # 通过好坏比例的比较，确定woe是否上升
																		           
    check_woe_asc    = True if woe_asc ==None else  cur_woe_asc == woe_asc         # 检查woe方向是否与预期一致
    woe_asc          = cur_woe_asc if woe_asc ==None else woe_asc                  # 首次woe方向为空，需要返回woe方向   
    cut_valid        = check_sample_num & check_woe_asc                            # 样本足够且woe方向正确，则本次切割有效
    return cut_valid,woe_asc

# 获取箱体切割点
def cutBin(bin_x,bin_y,woe_asc,min_sample):
    cutPoint = getKsCutPoint(bin_x,bin_y)                                          # 获取最大KS切割点
    is_cut_valid,woe_asc = checkCutValid(bin_x,bin_y,cutPoint,woe_asc,min_sample)  # 检查切割点是否有效
    if( not is_cut_valid):                                                         # 如果切割点无效
        cutPoint = None                                                            # 返回None
    return  cutPoint,woe_asc                                                       # 返回切割点

# 检查箱体是否不需再分
def checkBinFinish(y,min_sample):
    check_sample_num =  len(y)<min_sample                                          # 检查样本是否足够
    check_class_pure = (sum(y) == len(y))| (sum(y) == 0)                           # 检查样本是否全为一类
    bin_finish       =  check_sample_num | check_class_pure                        # 如果样本足够或者全为一类，则不需再分
    return bin_finish                                                              # 返回检查结果
																                   
# KS分箱主流程                                                                     
def ksMerge(x,y,bin_num=5,min_sample=None):           
    # -----初始化分箱列表等变量----------------                                    
    if min_sample==None:                                                           # 如果最小样本数不设置
        min_sample= np.floor((len(x)/bin_num)/3)                                   # 最小样本数采用该默认值
    un_cut_bins = [[min(x)-0.1,max(x)]]                                            # 初始化待分箱列表
    finish_bins = []                                                               # 初始化已完成分箱列表
    woe_asc     = None                                                             # 初始化woe方向
    # -----如果待分箱体不为空，则对待分箱进行分箱----------------                  
    for i in range(10000):                                                         # 为避免有bug使用while不安全，改为for           
        cur_bin = un_cut_bins.pop(0)                                               # 从待分箱列表获取一个分箱
        bin_x   = x[(x>cur_bin[0])&(x<=cur_bin[1])]                                # 当前分箱的x
        bin_y   = y[(x>cur_bin[0])&(x<=cur_bin[1])]                                # 当前分箱的y
        cutPoint,woe_asc = cutBin(bin_x,bin_y,woe_asc,min_sample)                  # 获取分箱的最大KS切割点
        if (cutPoint==None):                                                       # 如果切割点无效
            finish_bins.append(cur_bin)                                            # 将当前箱移到已完成列表
        else:                                                                      # 如果切割点有效
            # ------检查左箱是否需要再分,需要再分就添加到待分箱列表，否则添加到已完成列表-----
            left_bin    = [cur_bin[0],cutPoint]                                    # 生成左分箱
            left_y      = bin_y[bin_x <=cutPoint]                                  # 获取左箱y数据
            left_finish = checkBinFinish(left_y,min_sample)                        # 检查左箱是否不需再分
            if (left_finish):                                                      # 如果左箱不需再分
               finish_bins.append(left_bin)                                        # 将左箱添加到已完成列表
            else:                                                                  # 否则
               un_cut_bins.append(left_bin)                                        # 将左箱移到待分箱列表
               
             # ------检查右箱是否需要再分,需要再分就添加到待分箱列表，否则添加到已完成列表-----   
            right_bin    = [cutPoint,cur_bin[1]]                                   # 生成右分箱
            right_y      = bin_y[bin_x >cutPoint]                                  # 获取右箱y数据
            right_finish = checkBinFinish(right_y,min_sample)                      # 检查右箱是否不需再分
            if (right_finish):                                                     # 如果右箱不需再分
               finish_bins.append(right_bin)                                       # 将右箱添加到已完成列表
            else:                                                                  # 否则
               un_cut_bins.append(right_bin)                                       # 将右箱移到待分箱列表
        # 检查是否满足退出分箱条件：待分箱列表为空或者分箱数据足够
        if((len(un_cut_bins)==0)|((len(un_cut_bins)+len(finish_bins))>=bin_num)):  # 检查条件
            break                                                                  # 如果满足,则退出分箱
        
    # ------获取分箱切割点-------
    bins    = un_cut_bins + finish_bins                                            # 将完成或待分的分箱一起作为最后的分箱结果
    bin_cut = [cur_bin[1] for cur_bin in bins]                                     # 获取分箱右边的值
    list.sort(bin_cut)                                                             # 排序
    bin_cut.pop(-1)                                                                # 去掉最后一个，就是分箱切割点
    bin_cut = np.array(bin_cut)                                                    # 转换为numpy数据类型
    min_gap = min(bin_cut[1:]-bin_cut[:-1]) if len(bin_cut)>1 else 0               # 分箱之间的最小间隔
    if(abs(min_gap)>0.0001):                                                       # 如果间隔足够
        bin_cut = bin_cut.round(4)                                                 # 进行截断
    return list(bin_cut)                                                           # 返回结果
																                   
def ksMergeEnum(x,y,bin_num=5,min_sample=None):                                           
    # 转为badrate,再按连续变量进行分箱
    data = pd.DataFrame({'x':x,'_is_bad':y })                                     # 将x,y拼接
    df  = data.groupby(x)['_is_bad'].agg([('bad_cn','sum'),('cn','count')])       # 按组别统计坏样本个数与总样本个数
    df = pd.DataFrame({'grp':df.index,'bad_rate':df['bad_cn']/df['cn']})          # 计算bad_rate
    bad_rate_dict = dict(zip(df['grp'], df['bad_rate']))                          # 转为字典:组别与bad_rate
    x_br   = data['x'].map(bad_rate_dict)                                         # 将x转换为bad_rate数据
    bin_cut = ksMerge(x_br,y,bin_num,min_sample)                                  # 使用KS分箱进行分箱
																		          
    # 按照分割结果,找出每个切割点对应的枚举值                                     
    bin_set = [tuple(df.loc[df['bad_rate']<=bin_cut[0],'grp'])]                   # 第一个分箱的枚举值
    for i in range(len(bin_cut)-1):                                               # 逐个分箱拼接
        idx = ( df['bad_rate']>bin_cut[i])&( df['bad_rate']<=bin_cut[i+1])        # 第i+1个分箱的     
        bin_set.append(tuple(df.loc[idx,'grp']))                                  # 第i+1个分箱的枚举值 
    bin_set.append(tuple(df.loc[ df['bad_rate']>bin_cut[-1],'grp']))              # 拼接最后一个分箱
    return bin_set                                                                # 返回分箱结果