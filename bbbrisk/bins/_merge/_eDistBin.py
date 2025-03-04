import numpy as np
"""
本代码用于实现:等距分箱
本代码由《老饼讲解-机器学习》www.bbbdata.com编写
"""

'''在起始结束之间等距分箱'''
def _get_Ecut(start,end,cut_num):                        
    cut = np.linspace(start,end,cut_num+2)            # 在起始、结束生成cut_num个点
    cut = np.delete(cut,[0,-1]).tolist()              # 去掉起始、结束位置,就是切割点
    return cut

'''等距分箱'''
def eDistBin(x,bin_num=10):
    cut = []                                           # 初始化切割点
    if(isinstance(bin_num, (int, float))):             # 如果是数值,则直接切害
        cut = _get_Ecut(min(x),max(x),bin_num-1)       # 调用函数,对整体范围进行等距切割
    if(isinstance(bin_num, list)):                     # 如果是列表,则分段切割 
        last_end = float('inf')                        # 初始化上一次的切割结束点
        for seg in bin_num:                            # 逐段切割  
            start = seg[0]                             # 当前段的起始点
            end   = seg[1]                             # 当前段的结束点
            cut_num = seg[2]-1                         # 切点个数
            if(str(start)=='-'):                       # 如果起始是负号
                start = min(x)                         # 则以最小值作为起始点
            if(str(end)=='+'):                         # 如果结束点是正号
                end = max(x)                           # 则以最大值作为结束点
            cur_cut = _get_Ecut(start,end,cut_num)     # 对当前段进行切割   
            if(start>last_end):                        # 如果当前起始点大于上个结束点
                cut.append(start)                      # 则把当前起始点也作为一个切割点
            cut= cut+cur_cut                           # 添加本段的切割点
            cut.append(end)                            # 将结束点作为一个切割点
            last_end = end                             # 更新"上一次结束点"
        cut = np.delete(cut,-1)                        # 去掉最后一个切割点
    cut = np.array(cut)                                # 转换为numpy数据类型
    min_gap = min(cut[1:]-cut[:-1])                    # 分箱之间的最小间隔
    if(abs(min_gap)>0.0001):                           # 如果间隔足够
        cut = cut.round(4)                             # 进行截断
    return cut                                         # 返回切割点
