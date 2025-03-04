# -*- coding: utf-8 -*-
"""
用于打印相关报告
"""
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt 

''' 计算阈值表 '''
def get_threshold_tb(score,y,bin_step=10):
    # 计算分箱起始结束范围
    bin_start = math.trunc(score.min()/bin_step)*bin_step                                     # 计算分箱起始值
    bin_end   = math.trunc(score.max()/bin_step+1)*bin_step                                   # 计算分箱结束值
    tb        = pd.DataFrame(columns=['分组名称','本组客户','本组好客户','本组坏客户'])       # 初始化阈值表         
																				              
    # 统计分组内的好坏客户个数                                                                
    for cur_bin in range(bin_start,bin_end,bin_step):                                         # 逐组统计
        bin_name ='['+str(cur_bin)+'-'+str(cur_bin+bin_step)+')'                              # 当前分箱的名称
        cur_y    = y[(score>=cur_bin)&(score<cur_bin+bin_step)]                               # 获取当前分箱的y
        cn       = cur_y.shape[0]                                                             # 当前分箱的总客户个数
        bad_cn   = cur_y.sum()                                                                # 当前分箱的坏客户个数
        good_cn  = cn-bad_cn                                                                  # 当前分箱的好客户个数
        tb.loc[tb.shape[0]]=[bin_name,cn,good_cn,bad_cn]                                      # 记录到阈值表中
																							  
    #计算阈值表其它字段                                                                       
    tb['总客户']       = tb['本组客户'].sum()                                                  # 总客户数
    tb['总好客户']     = tb['本组好客户'].sum()                                                # 总好客户数
    tb['总坏客户']     = tb['本组坏客户'].sum()                                                # 总坏客户数
    tb['阈值']         = tb['分组名称'].apply(lambda x:'<'+x.split('-')[1].replace(')',''))    # 从分组名称中获取本组阈值
    tb['损失客户']     = tb['本组客户'].cumsum()                                               # 损失客户个数
    tb['损失客户%']    = tb['损失客户']/tb['总客户']                                           # 损失客户占比
    tb['损失好客户']   = tb['本组好客户'].cumsum()                                             # 损失的好客户
    tb['损失好客户%']  = tb['损失好客户']/tb['总好客户']                                       # 损失的好客户占比
    tb['剔除坏客户']   = tb['本组坏客户'].cumsum()                                             # 剔除的坏客户
    tb['剔除坏客户%']  = tb['剔除坏客户']/tb['总坏客户']                                       # 剔除的坏客户占比
    tb['本组坏客户占比']  = tb['本组坏客户']/np.maximum(tb['本组客户'],1)                      # 本组坏客户占比
    tb['损失客户中坏客户占比'] = tb['剔除坏客户']/tb['损失客户']                               # 损失客户中坏客户的占比
    tb = tb.drop(['总客户','总好客户','总坏客户'],axis=1)                                      # 删除不要的列
    return tb                                                                                  # 返回阈值表

''' 画出分数分布图 '''
def draw_score_disb(score,y,bin_step=10,figsize=(12, 4)):                                              
    tb = get_threshold_tb(score,y,bin_step)                                                             # 获取阈值表
    x_axis = tb['分组名称'].apply(lambda x: x.split('-')[0].replace('[',''))                            # 分组的名称
    plt.bar(x_axis, tb['本组好客户'], align="center", color="#66c2a5",label="good")                     # 好客户柱状图
    plt.bar(x_axis, tb['本组坏客户'], align="center", bottom=tb['本组好客户'], color='r', label="bad")  # 坏客户柱状图 
    plt.rcParams["figure.figsize"] = figsize                                                            # 设置figure_size尺寸
    plt.legend()                                                                                        # 显示图例
    plt.show()                                                                                          # 显示图片