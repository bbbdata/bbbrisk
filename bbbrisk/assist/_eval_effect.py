from sklearn import metrics
import pandas as pd

def eval_best_effect(x,y):
    print('开始评估...')                                                           # 打印进度
	
	# ----统计样本键值的好坏个数-----
    stat_dict  = {}                                                                # 统计数据(字典格式)
    sample_key = []                                                                # 样本的键值
    m = x.shape[0]                                                                 # 样本个数
    for i in range(m):                                                             # 逐个样本循环
        row_str = x.iloc[i].astype(str).str.cat(sep=',')                           # 将当前样本合并为字符,作为键值
        sample_key.append(row_str)                                                 # 记录当前样本的键值
        if row_str not in stat_dict:                                               # 如果当前键值未被统计过
           stat_dict[row_str] =[0,0]                                               # 初始化当前键值好、坏个数都为0
        stat_dict[row_str][y[i]] +=1                                               # 更新当前键值的统计个数
        if(i%1000==0):                                                             # 是否打印进度
            print('当前进度：',"{:.2f}".format(i/m*100),'%')                       # 打印进度
	
	# ----样本键值统计表：好坏个数、占比-----															                  
    keys = stat_dict.keys()                                                        # 所有键值
    val  = pd.DataFrame(stat_dict.values())                                        # 键的好坏个数
    p    = val[1]/(val[0]+val[1])                                                  # 该键是坏样本的概率
    stat_tb =  pd.DataFrame({'key':keys,'good_cn':val[0],'bad_cn':val[1],'p':p})   # 记录统计表
    
	# ---计算样本是坏样本的概率------
    p_dict={}                                                                      # 各个键是坏样本的概率(字典格式)
    for key, val in stat_dict.items():                                             # 逐键计算
        p_dict[key] = val[1]/sum(val)                                              # 记录当前键是坏样本的概率
    p = [p_dict[key]  for key in sample_key]                                       # 根据键值,获取样本是坏样本的概率
    
	# ---- 计算AUC、KS--------------------
    fpr, tpr, thresholds= metrics.roc_curve(y,p)                                   # 计算fpr, tpr
    ks  = max(abs(fpr-tpr))                                                        # 计算KS: abs(fpr-tpr)最大者就是KS
    auc = metrics.auc(fpr, tpr)                                                    # 计算AUC
    print('...完成评估!')                                                          # 打印进度
    return auc,ks,p,stat_tb                                                        # 返回结果
