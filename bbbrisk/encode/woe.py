# -*- coding: utf-8 -*-
"""
本代码用于woe相关的处理
"""
import pandas as pd
import numpy as np

'''用于计算单个变量的woe'''  
def _get_one_woe(pair):                                                                                                        
    col_name = pair.columns.tolist()                                                      # 获取变量名
    x,y = col_name[0],col_name[1]                                                         # 第1列是x,第2列是y
    bad_tt  = pair[y].sum()                                                               # 坏客户个数
    good_tt = pair.shape[0] -bad_tt                                                       # 好客户个数
    woe_df  = pair.groupby(x)[y].agg([('bad_cn','sum'),('cn','count')])                   # 按组别统计坏客户个数与总客户个数
    null_cn = pair[x].isnull().sum()                                                      # 空值的个数
    if(null_cn>0):                                                                        # 如果有空值
        null_bad_cn = pair.loc[pair[x].isnull(),y].sum()                                  # 空值的坏客户个数
        nullStat = pd.DataFrame({'bad_cn':null_bad_cn ,'cn':null_cn},pd.Index([None]))    # 空值的统计信息
        woe_df = pd.concat([woe_df,nullStat],axis=0)                                      # 将空值的统计信息拼接到总表
    woe_df['good_cn'] = woe_df['cn'] -woe_df['bad_cn']                                    # 好客户个数
    woe_df['bad_cn']  = np.maximum(woe_df['bad_cn'],1)                                    # 如果坏客户个数为0,则设为1.这样计算woe才不报错
    woe_df['good_cn'] = np.maximum(woe_df['good_cn'],1)                                   # 如果好客户个数为0,则设为1.这样计算woe才不报错
    woe_df['woe']     = np.log((woe_df['bad_cn']/bad_tt)/(woe_df['good_cn']/good_tt))     # 按公式计算woe
    woe_df['grp']     = woe_df.index                                                      # 添加组别号  
    woe_df['var']     = x                                                                 # 添加变量名
    woe_tb= woe_df[['var','grp','woe']]
    woe_dict = {item:woe_df['woe'][item] for item in woe_df.index}                        # 整理为字典格式
    return woe_dict,woe_tb                                                                # 返回woe表格

'''用于计算多个变量的woe'''   
def get_woe_dict(x,y):                                                                         
    woe_dict={}                                                                           # 初始化woe字典
    var_list = x.columns.tolist()
    for var in var_list:                                                                  # 逐个变量(即每列)循环
        data = pd.DataFrame({var:x[var],'_is_bad':y })                                    # 将当前变量与y拼接
        cur_woe_dict,cur_woe_tb =  _get_one_woe(data)                                     # 计算当前变量的woe
        woe_dict[var] = cur_woe_dict                                                      # 将当前变量的woe字典添加到woe总字典
    return woe_dict

'''将woe字典转为表格'''   
def woe_dict_to_tb(woe_dict):
    woe_tb = pd.DataFrame()                                                               # 初始化woe表
    for var, woe in woe_dict.items():                                                     # 逐个变量循环
        cur_woe_tb = pd.DataFrame({'var':var,'grp':woe.keys(),'woe':woe.values() })       # 当前变量的woe信息
        woe_tb = cur_woe_tb if woe_tb.empty else pd.concat([woe_tb,cur_woe_tb],axis=0)    # 将当前变量的woe表添加到woe总表
    return woe_tb

'''将数据转换为woe '''                                                                                      
def to_woe(x,y):                                                                      
    woe_data = x.copy()                                                                   # 先将数据复制一份
    is_dict = isinstance(y, dict)                                                         # 判断是否字典类型
    if(is_dict):                                                                          # 如果是字典
        woe_dict = y                                                                      # y就是woe映射表
    else:                                                                                 # 否则,y是样本标签
        woe_dict  = get_woe_dict(x,y)                                                     # 计算woe映射表
    for col, row in woe_data.items():                                                     # 逐个变量(即每列)循环
            woe_data[col] = woe_data[col].map(woe_dict[col])                              # 将变量组别转换为woe值
    if(is_dict):                                                                          # 如果y是woe字典
        return woe_data                                                                   # 则只返回转换后的woe数据
    return woe_data,woe_dict                                                              # 返回转换后的woe数据,以及woe映射表