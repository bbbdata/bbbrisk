# -*- coding: utf-8 -*-
"""
"""
import numpy as np
import pandas as pd

# 分箱类
class Bins:
    def __init__(self,bin_set,with_null=False,with_other=False):
        self._esp = 0.1**15                                                          # 一个极小的数
		# 添加None和other分箱
        bin_set    = [ b if  isinstance(b, tuple) else (b,) for b in bin_set]        # 将所有分箱标准化:转换为元组格式
        has_null  = False                                                            # bin_set设置里是否包含None配置
        has_other = False                                                            # bin_set设置里是否包含'other'配置    
        for b in bin_set:                                                            # 历遍分箱
            for seg in b:                                                            # 历遍分箱中的段
                if (seg==None):                                                      # 如果该段为None
                   has_null = True                                                   # 则标记有None配置
                if (seg=='_other'):                                                  # 如果该段为_other
                   has_other = True                                                  # 标记有other配置
        if((with_null==True)& (has_null==False)):                                    # 如果需要None,而没有None配置
            bin_set.append((None,))                                                  # 则添加None配置
        if((with_other==True)& (has_other==False)):                                  # 如果需要other,而没有other配置
            bin_set.append(('_other',))                                              # 则添加other配置
			
		# 记录分箱配置、分箱范围与说明
        bin_range,bin_desc = self._getBinRange(bin_set)                              # 获取配置的分箱范围和说明
        self.bin_set   = bin_set                                                     # 记录分箱配置
        self.bin_range = bin_range                                                   # 记录范围
        self.bin_desc  = bin_desc                                                    # 记录分箱描述
        self.bin_desc_dict  = bin_desc.to_dict()['desc']                             # 分箱描述字典格式

    '''获取一个段的范围与描述'''
    def _get_seg_range(self,seg):
        # 共三种类型:枚举值,预留值_other,和范围
        # 范围的配置(范围原则上默认左开右闭):
        # [0,2],['-',2],[2,'+']:分别代表(0,2],<=2,>2
        # [0,2,'[)']、[0,2,')']、[2,'+','[']:则第三个原素指定了开闭
        # 关于seg_range: 这里的范围seg_range=[x1,x2]代表的是(x1,x2]
        if(not isinstance(seg, list)):                                               # 如果不是list类型(枚举值或other) 
            seg_desc  = str(seg)                                                     # 则描述直接用自身
            seg_range = seg                                                          # 范围也是自身
        elif((str(seg[0])=='-')&(str(seg[1])=='+')):                                 # 如果为['-','+']
            seg_desc  = 'all'                                                        # 描述为'all'
            seg_range = [float('-inf'),float('inf')]                                 # 范围为[-inf,inf]
        elif(str(seg[0])=='-'):                                                      # 如果左边是-
            seg_desc  = '<='+str(seg[1])                                             # 则描述为<=x2
            seg_range = [float('-inf'),seg[1]]                                       # 范围取为(-inf,x2]
            if(len(seg)==3):                                                         # 如果有范围配置
                if (seg[2] == ')'):                                                  # 如果配为)
                    seg_desc  = '<'+str(seg[1])                                      # 则描述为<x2
                    seg_range = [float('-inf'),seg[1]- self._esp]                    # 范围取为(-inf,x2-esp]
        elif(str(seg[1])=='+'):                                                      # 如果右边为'+'
            seg_desc  = '>'+str(seg[0])                                              # 则描述为>=x1
            seg_range = [seg[0],float('inf')]                                        # 范围取为(x1,inf]
            if(len(seg)==3):                                                         # 如果有范围配置
                if (seg[2] == '['):                                                  # 如果配为[
                    seg_desc  = '>='+str(seg[0])                                     # 则描述为>=x2
                    seg_range = [seg[0]- self._esp,float('inf')]                     # 范围取为(-inf,x2-esp]
        else:                                                                        # 配置了左右值的范围处理如下
             seg_desc  = '('+str(seg[0])+','+str(seg[1])+']'                         # 默认的描述(左开右闭)
             seg_range = seg                                                         # 默认的取值范围(左开右闭)
             if(len(seg)==3):                                                        # 如果有范围配置
                seg_desc_left  = '('+str(seg[0])                                     # 默认的左描述(左开)
                seg_range_left = [seg[0]]                                            # 默认的左范围(左开)
                if(seg[2][0] == '['):                                                # 如果配为左闭
                    seg_desc_left  = '['+str(seg[0])                                 # 则左描述为左闭
                    seg_range_left = [seg[0]-self._esp]                              # 左范围配为:(x1-esp
                seg_desc_right  = str(seg[1])+']'                                    # 默认右范围描述为右闭
                seg_range_right = [seg[1]]                                           # 默认右范围为:x2]
                if(seg[2][1] == ')'):                                                # 如果右范围配为开区间
                    seg_desc_right  = str(seg[1])+')'                                # 则右范围描述改为右开
                    seg_range_right = [seg[0]-self._esp]                             # 右范围改为:x2-esp]
                seg_desc  =  seg_desc_left+','+seg_desc_right                        # 整体范围描述
                seg_range =  seg_range_left+seg_range_right                          # 整体范围
        return seg_range,seg_desc                                                    # 返回范围与范围描述

    '''获取分箱的范围与描述'''
    def _getBinRange(self,bin_set):
        bin_desc  = []                                                               # 初始化范围描述
        bin_range = []                                                               # 初始化范围
        for b in bin_set:                                                            # 逐箱循环
            seg_range = [self._get_seg_range(seg)[0] for seg in b]                   # 获取当前箱的范围
            seg_desc  = [self._get_seg_range(seg)[1] for seg in b]                   # 获取当前箱的范围描述
            bin_range.append(seg_range)                                              # 添加当前箱的范围
            bin_desc.append(' & '.join(seg_desc))                                    # 添加当前箱的范围描述
        bin_desc  = pd.DataFrame({'desc':bin_desc})                                  # 将范围描述转为df
        bin_range = dict(zip([i for i in range(len(bin_range))],bin_range))          # 将范围转为字典{组号:范围}
        return bin_range,bin_desc                                                    # 返回范围与范围描述
																				     
    '''将数据转换为箱号'''                                                           
    def _apply(self,x):                                                               
        b = np.full(len(x), np.nan)                                                  # 初始化箱号
        null_bin  = None                                                             # 空值所在箱号
        other_bin = None                                                             # other所在箱号
        for key, value in self.bin_range.items():                                    # 循箱循环
            for seg in value:                                                        # 逐段循环
                if(isinstance(seg, list)):                                           # 如果段是list
                    b[(x>seg[0]) & (x<=seg[1])]=key                                  # 则将x在段范围(左开右闭)内的位置标为组号
                elif(seg==None):                                                     # 如果段是None
                    null_bin = key                                                   # 记录None所属组号
                elif(seg=='_other'):                                                 # 如果段是other
                    other_bin = key                                                  # 记录other所属组号
                else:                                                                # 其它情况(即段是枚举值)
                    b[x==seg] = key                                                  # 则将x为该枚举的位置标为组号
        if (null_bin !=None):                                                        # 如果None有箱号
            b[pd.isnull(x)] = null_bin                                               # 则将x的空值位置标为组号
        if (other_bin !=None):                                                       # 如果other有箱号
            b[pd.isnull(b)] = other_bin                                              # 则将b中剩余取值都标为other箱号
        return b                                                                     # 返回组号数据 

    '''计算分箱详情'''
    def binStat(self,x,y):
        grp      = np.arange(len(self.bin_desc))                                     # 组号
        bin_stat  = pd.DataFrame({'grp':grp,'grp_desc':self.bin_desc['desc']})       # 初始化分箱统计
        x_grp = self._apply(x)                                                       # 将x转为组号
        for i in range(bin_stat.shape[0]):                                           # 逐组统计
            bin_stat.loc[i,'cn']      = int((x_grp==i).sum())                        # 统计样本个数
            bin_stat.loc[i,'good_cn'] = ((x_grp==i)&(y==0)).sum()                    # 统计好样本个数
            bin_stat.loc[i,'bad_cn']  = ((x_grp==i)&(y==1)).sum()                    # 统计坏样本个数
        bin_stat['bad_rate'] = bin_stat['bad_cn']/np.maximum(bin_stat['cn'],1)       # 坏样本占比
																					 
        # 计算IV值                                                                  
        bad_tt    = y.sum()                                                          # 总的坏样本个数
        good_tt   = len(y) -bad_tt                                                   # 总的好样本个数
        bad_rate  = np.maximum(bin_stat['bad_cn'],1)/bad_tt                          # 每组坏样本占比
        good_rate = np.maximum(bin_stat['good_cn'],1)/good_tt                        # 每组好样本占比
        bin_stat['iv']  = (bad_rate-good_rate)*np.log(bad_rate/good_rate)            # 计算iv
        bin_stat.loc[bin_stat['cn']==0,'iv']=0                                      
        # 统计汇总信息                                                              
        s = bin_stat.sum()                                                           # 对各组进行求和
        totalStat = pd.DataFrame({'grp':'sum'                                       
                      ,'grp_desc':'总计'                                            
                      ,'cn':s['cn']                                                 
                      ,'good_cn':s['good_cn']                                       
                      ,'bad_cn':s['bad_cn']                                         
                      ,'bad_rate':s['bad_cn']/s['cn']                               
                      ,'iv':s['iv']}                                                
                      ,pd.Index([bin_stat.shape[0]]))                                
        bin_stat = pd.concat([bin_stat,totalStat],axis=0)                            # 拼接汇总信息
        bin_stat['cn']      = bin_stat['cn'].astype(int)                             # 修改cn的数据类型
        bin_stat['bad_cn']  = bin_stat['bad_cn'].astype(int)                         # 修改bad_cn的数据类型
        bin_stat['good_cn'] = bin_stat['good_cn'].astype(int)                        # 修改good_cn的数据类型
        return bin_stat[['grp','grp_desc','cn','bad_cn','good_cn','iv','bad_rate']]  # 返回分箱统计结果

