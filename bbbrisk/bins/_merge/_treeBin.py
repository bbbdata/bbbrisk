from sklearn import tree
import pandas as pd
import numpy as np


# 决策树分箱-连续变量  
def treeMerge(x,y,max_depth=3,min_sample=None):   
    min_sample= int(np.floor((len(x)/5)/3))
    clf     = tree.DecisionTreeClassifier(min_samples_leaf=min_sample,max_depth=max_depth)  # sk-learn的决策树模型
    clf     = clf.fit(pd.DataFrame(x), y)                                                   # 用数据训练树模型
    bin_cut = clf.tree_.threshold[clf.tree_.children_left>0]                                # 提取切割点
    bin_cut.sort()                                                                          # 切割点进行排序
    min_gap = min(bin_cut[1:]-bin_cut[:-1]) if len(bin_cut)>1 else 0                        # 分箱之间的最小间隔
    if(abs(min_gap)>0.0001):                                                                # 如果间隔足够
        bin_cut = bin_cut.round(4)                                                          # 进行截断
    return list(bin_cut)                                                                    # 返回结果


# 决策树分箱-枚举变量    
def treeMergeEnum(x,y,max_depth=3,min_sample=None):                                    
    # 转为badrate,再按连续变量进行分箱
    data = pd.DataFrame({'x':x,'_is_bad':y })                                               # 将x,y拼接
    df   = data.groupby(x)['_is_bad'].agg([('bad_cn','sum'),('cn','count')])                # 按组别统计坏样本个数与总样本个数
    df   = pd.DataFrame({'grp':df.index,'bad_rate':df['bad_cn']/df['cn']})                  # 计算bad_rate
    bad_rate_dict = dict(zip(df['grp'], df['bad_rate']))                                    # 转为字典:组别与bad_rate
    x_br    = data['x'].map(bad_rate_dict)                                                  # 将x转换为bad_rate数据
    bin_cut = treeMerge(x_br,y,max_depth,min_sample)                                        # 使用KS分箱进行分箱
																		                    
    # 按照分割结果,找出每个切割点对应的枚举值                                               
    bin_set = [tuple(df.loc[df['bad_rate']<=bin_cut[0],'grp'])]                             # 第一个分箱的枚举值
    for i in range(len(bin_cut)-1):                                                         # 逐个分箱拼接
        idx = ( df['bad_rate']>bin_cut[i])&( df['bad_rate']<=bin_cut[i+1])                  # 第i+1个分箱的     
        bin_set.append(tuple(df.loc[idx,'grp']))                                            # 第i+1个分箱的枚举值 
    bin_set.append(tuple(df.loc[ df['bad_rate']>bin_cut[-1],'grp']))                        # 拼接最后一个分箱
    return bin_set                                                                          # 返回分箱结果
