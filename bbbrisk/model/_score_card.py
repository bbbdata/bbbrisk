# -*- coding: utf-8 -*-
"""

"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

from  .. import bins
from  .. import encode

''' 评分卡类 '''
class _DataTransfer:
    def __init__(self,x,y,bin_sets):
        # 存储分箱转换对象
        bin_obj_dict  = {}                                                                                   # 初始化分箱对象字典
        bin_desc_dict = {}                                                                                   # 初始化箱号描述字典
        for var, bin_set in bin_sets.items():                                                                # 逐变量循环
            bin_obj_dict[var]  = bins.Bins(bin_set)                                                          # 当前变量的分箱类对象
            bin_desc_dict[var] = bin_obj_dict[var].bin_desc_dict                                             # 记录当前变量的箱号描述
        self._bin_obj_dict  = bin_obj_dict                                                                   # 记录变量的分箱对象字典
        self._bin_desc_dict = bin_desc_dict                                                                  # 记录变量的箱号描述字典
        
        # woe转换字典
        x_grp    = self.to_grp(x)                                                                             # 将数据转换为分箱    
        woe_dict = encode.woe.get_woe_dict(x_grp,y)                                                           # 获取woe映射字典
        self._woe_dict = woe_dict                                                                             # 记录woe映射字典
    def _keep(self,var_list):
        bin_obj_dict  = {}                                                                                    # 初始化分箱对象字典
        bin_desc_dict = {}                                                                                    # 初始化箱号描述字典
        woe_dict      = {}                                                                                    # 初始化woe映射字典
        for var in var_list:                                                                                  # 逐变量循环
           bin_obj_dict[var]  = self._bin_obj_dict[var]                                                       # 当前变量的分箱对象字典
           bin_desc_dict[var] = self._bin_desc_dict[var]                                                      # 当前变量的箱号描述字典
           woe_dict[var]      = self._woe_dict[var]                                                           # 当前变量的woe映射字典
        self._bin_obj_dict  = bin_obj_dict                                                                    # 记录变量的分箱对象字典
        self._bin_desc_dict = bin_desc_dict                                                                   # 记录变量的箱号描述字典
        self._woe_dict      = woe_dict                                                                        # 记录woe映射字典
        return self
 
        
	# 将x转换为分箱数据  
    def to_grp(self,x):
        x_grp    = pd.DataFrame(columns=x.columns)                                                            # 初始化转换后的箱号数据
        for i in range(x.shape[1]):                                                                           # 逐个变量自动分箱
            var        = x.columns[i]                                                                         # 当前变量的名称
            cur_x      = x.iloc[:,i]                                                                          # 变量的数据
            b          = self._bin_obj_dict[var]                                                              # 当前的分箱配置
            x_grp[var] = b._apply(cur_x)                                                                      # 将x转换为箱号
        return x_grp                                                                                          # 返回分组数据
		
	# 将x转换为woe数据		
    def to_woe(self,x):
        woe_data = self.to_grp(x)                                                                             # 先将数据转为分组数据
        for col, row in woe_data.items():                                                                     # 逐个变量(即每列)循环
            woe_data[col] = woe_data[col].map(self._woe_dict[col])                                            # 将变量组别转换为woe值
        return woe_data                                                                                       # 返回转换后的woe数据
        
        
''' 评分卡类 '''
class _Card:
    def __init__(self,data_transfer,param_dict):

        var = list(param_dict['w'].keys())
																	                                          
        # 评分卡的模型与分箱信息                                                                              
        self._param_dict       = param_dict                                                                   # 模型的参数(字典格式)
        self._data_transfer    = data_transfer._keep(var)                                                     # 数据转换器
																	                                          												                                          
        # 模型转评分参数与结果                                                                                                  
        self.var               = var                                                                          # 评分卡所用的变量
        self.d                 = None                                                                         # 模型odds为d时
        self.B                 = None                                                                         # 评分为B
        self.k                 = None                                                                         # odds每降k倍
        self.S                 = None                                                                         # 分数提升S分
        self.factor            = None                                                                         # 模型转评分时使用的factor
        self.offset            = None                                                                         # 模型转评分时使用的offset
        self.baseScore         = None                                                                         # 基础分
        self.featureScore      = None                                                                         # 特征得分
        self.featureScore_dict = None                                                                         # 特征得分(字典格式)
																						                      
                                                            
    def build(self,d=50,B=600,k=2,S=20):                                                                      
        # ---预读取变量---                                                                                    
        param_dict    = self._param_dict                                                                      # 读取模型的参数
        bin_desc_dict = self._data_transfer._bin_desc_dict                                                    # 读取分箱描述
        woe_dict      = self._data_transfer._woe_dict                                                         # 读取模型的x
																						                      
        # ---计算得分表---                                                                                    
        factor    = -S/(np.log(k))                                                                            # 计算factor
        offset    = B-factor*np.log(d)                                                                        # 计算offset
        baseScore = offset+ factor*param_dict['b']                                                            # 计算基础分
        featureScore = pd.DataFrame()                                                                         # 初始化特征得分表
        featureScore_dict = {}                                                                                # 初始化特征得分字典
        for var, w in param_dict['w'].items():                                                                # 对x字典历遍
            grp   = bin_desc_dict[var].keys()                                                                 # 组号
            desc  = bin_desc_dict[var].values()                                                               # 分组描述
            val   = woe_dict[var].values()                                                                    # 分组x的值
            score = factor*w*np.array(list(val))                                                              # 分组得分
            var_info = pd.DataFrame({'var':var,'grp':grp,'desc':desc,'woe':val,'w':w,'score':score})          # 当前变量的分组信息
            featureScore = var_info if featureScore.empty else pd.concat([featureScore,var_info])             # 添加到特征得分表中
            featureScore_dict[var] = dict(zip(grp, score))                                                    # 添加到特征得分字典
																						                      
        # ---记录变量---                                                                                      
        self.d                 = d                                                                            # 模型odds为d时
        self.B                 = B                                                                            # 评分为B
        self.k                 = k                                                                            # odds每降k倍
        self.S                 = S                                                                            # 分数提升S分
        self.factor            = factor                                                                       # 记录factor
        self.offset            = offset                                                                       # 记录offset
        self.baseScore         = baseScore                                                                    # 记录基础分
        self.featureScore      = featureScore                                                                 # 记录特征得分
        self.featureScore_dict = featureScore_dict                                                            # 特征得分(字典形式)
        return self                                                                                           
																						                      
																						                      
    '''模型转评分(通过分数范围来生成评分卡)'''                                                                
    def build_with_range(self,score_min,score_max,d=50,k=2):                                             
        param_dict  = self._param_dict                                                                        # 读取模型的参数
        woe_dict    = self._data_transfer._woe_dict                                                           # 读取woe映射表,它是逻辑回归模型的输入
		# 计算wx+b的最大、最小值                                                                              
        ln_odds_min = param_dict['b']                                                                         # 初始化ln_odds(即wx+b)的最小值为b
        ln_odds_max = param_dict['b']                                                                         # 初始化ln_odds(即wx+b)的最大值为b
        for var, w in param_dict['w'].items():                                                                # 对x字典历遍
            val = woe_dict[var].values()                                                                      # 当前变量的woe值
            ln_odds_min = ln_odds_min + w*min(list(val))                                                      # 将最小值加到ln_odds_min中
            ln_odds_max = ln_odds_max + w*max(list(val))                                                      # 将最大值加到ln_odds_max中
																						                      
		# 计算S和B                                                                                            
        factor = (score_min - score_max)/(ln_odds_max - ln_odds_min)                                          # 计算factor
        offset = score_max -factor*ln_odds_min                                                                # 计算offset
        S      = -np.log(k)*factor                                                                            # 计算S
        B      = offset+factor*np.log(d)                                                                      # 计算B
        self.build(d,B,k,S)                                                                                   # 重新生成评分卡表
																						                      
    '''评分预测'''                                                                                            
    def predict(self,x,with_item=False):                                                                      
        x    = x[self.var]                                                                                    # 评分所使用的变量
        x_grp = self._data_transfer.to_grp(x)                                                                 # 将x转换为分箱数据         
															                      
		# 计算评分                                                                                            
        score_item = x_grp                                                                                    # 初始化得分
        baseScore  = self.baseScore                                                                           # 基础得分
        featureScore_dict = self.featureScore_dict                                                            # 特征得分字典
        for var, cur_dict in featureScore_dict.items():                                                       # 逐个变量(即每列)循环
           score_item[var] = score_item[var].map(cur_dict)                                                    # 将变量组别转换为特征得分
        score_item['_baseScore'] = baseScore                                                                  # 基础分
        score = score_item.sum(axis=1)                                                                        # 计算总分
        if(with_item==True):                                                                                  # 如果需要得分明细 
            return score,score_item                                                                           # 则返回得分与明细
        return score                                                                                          # 否则只返回得分
																						              

''' 逻辑回归模型类 '''
class _LogitModel:
    def __init__(self,data_transfer,test_size = 0.2,penalty='l2',select_var=True,select_tol=0.005,random_state=None):
        
        # 用于训练的模型
        self.clf = LogisticRegression(penalty=penalty, dual=False, tol=0.0001, C=1.0
		                              ,fit_intercept=True, intercept_scaling=1
									  ,class_weight=None, random_state=random_state
                                      ,solver='lbfgs', max_iter=1000
									  ,verbose=0,warm_start=False,n_jobs=None,l1_ratio=None)    
		# 需要设置的参数                                                            
        self._data_transfer = data_transfer                                                                     # 数据转换对象
        self._test_size     = test_size                                                                         # 测试数据的比例
        self._select_var    = select_var                                                                        # 建模前是否选择变量
        self._select_tol    = select_tol                                                                        # 逐步向前选择变量时,AUC的最低提升比
        self._random_state  = random_state                                                                      # 随机种子,逻辑回归模型与数据分割两处用到
																						                      
		# 模型信息                                                                                                  
        self.var            = None                                                                              # 入模变量
        self.param_dict     = None                                                                              # 模型参数,字典格式
        self.w              = None                                                                              # 模型权重
        self.b              = None                                                                              # 模型阈值
        self.w_norm         = None                                                                              # 模型归一化训练时的权重
        self.b_norm         = None                                                                              # 模型归一化训练时的阈值
																						                      
        # 评估指标                                                                                             
        self.train_auc      = None                                                                              # 训练样本的AUC
        self.train_ks       = None                                                                              # 训练样本的KS
        self.test_auc       = None                                                                              # 测试样本的AUC
        self.test_ks        = None                                                                              # 测试样本的KS
																						                      
    '''模型的评估指标函数'''                                                                                  
    def _performance(self,clf,x,y,target='auc'):                                                                 
        pred_prob_y = clf.predict_proba(x)[:,1]                                                                # 预测概率
        fpr, tpr, thresholds= metrics.roc_curve(y,pred_prob_y)                                                 # 计算fpr, tpr
        ks  = max(abs(fpr-tpr))                                                                                # 计算KS: abs(fpr-tpr)最大者就是KS
        auc = metrics.auc(fpr, tpr)                                                                            # 计算AUC
        if target=='auc':                                                                                      # 如果只要auc
            return auc                                                                                         # 则只返回AUC
        return auc,ks                                                                                          # 否则返回auc和ks		
																						                      
    '''逐步回归-向前选择(用于选择变量)'''                                                                      
    def _stepwise(self,clf,X,y,perf_fcn,tol=0.005):                                                                 
        var_names  = X.columns                                                                                 # 变量名称
        select_var = []                                                                                        # 已挑选的变量  
        var_pool   = np.arange(X.shape[1])                                                                     # 待挑选变量池
        perf_rec   = []                                                                                        # 用于记录评估指标
        print("开始逐回步归...")                                                                              
        while(len(var_pool)>0):                                                                                # 如果还有变量,则继续挑选变量
            best_perf = float('-inf')                                                                          # 初始化本轮最佳评估指标     
            best_var  = None                                                                                   # 初始化本轮最佳变量
            #---选出剩余变量中能带来最好效果的变量--------                                                        
            for i in  var_pool:                                                                                # 逐个变量评估"加入该变量的效果"
                cur_x = X.iloc[:,select_var+[i]]                                                               # 新变量和已选变量作为建模数据  
                clf.fit(cur_x,y)                                                                               # 训练模型
                cur_perf = perf_fcn(clf,cur_x,y)                                                               # 当前的模型性能评估
                if(cur_perf>best_perf):                                                                        # 如果当前变量的评估指标更好
                    best_perf = cur_perf                                                                       # 将当前评估指标作为最佳评估指标
                    best_var  = i                                                                              # 将当前变量作为最佳变量
            # 如果加入本次变量有显著效果，则将该变量添加到已选变量                                             
            valid = True                                                                                       # 变量是否有效
            if(len(perf_rec)>0):                                                                               # 从第二个变量开始,检验添加变量是否有效
                valid =(best_perf-perf_rec[-1])/perf_rec[-1]>tol                                               # 加入本次变量是否效果显著 
            print("本轮最佳评估:",best_perf,",本轮最佳变量：",var_names[best_var])                             # 打印本轮的最佳评估指标和最佳变量
            if(valid):                                                                                                                
                perf_rec.append(best_perf)                                                                     # 记录本轮评估指标
                select_var.append(best_var)                                                                    # 将本次选择的变量加入"已选变量池"
                var_pool = var_pool[var_pool!=best_var]                                                        # 删除待选变量池
            else:                                                                                              # 如果效果不显著,就不再添加变量
                print('提升效果不明显,不添加该变量,完成变量选择')                                              # 打印提示
                break                                                                                          # 停止变量选择
        return select_var,var_names[select_var].to_list()                                                      # 返回选择的变量索引以及变量名称
		
    def _fit(self,x,y):
        print('\n建模开始........\n')
        x = self._data_transfer.to_woe(x)                                                                       # 将数据转换为woe
        # -----------模型训练-----------------------                                                              
        # 将数据归一化并进行分割                                                                                  
        x_min   = x.min(axis=0)                                                                                 # woe的最小值
        x_max   = x.max(axis=0)                                                                                 # woe的最大值
        x_norm  = (x-x_min)/(x_max-x_min)                                                                       # 将woe归一化  
        x_train,x_test,y_train,y_test = train_test_split(x_norm,y                                               
                                                         ,test_size=self._test_size                             
                                                         ,random_state=self._random_state)                      # 将数据分为训练数据与测试数据            
																											    
        # 变量选择与模型训练                                                                                    
        md_var_idx = np.arange(x_train.shape[1]).tolist()                                                       # 默认所有变量都入模
        md_var     = x_train.columns.to_list()                                                                  # 入模变量的名称
        if self._select_var == True:                                                                            # 如果设置选择变量
            md_var_idx,md_var= self._stepwise(self.clf,x_train,y_train,self._performance,tol=self._select_tol)  # 用逐步回归挑选变量
            print("最终选用变量",len(md_var),"个：",list(md_var))                                               # 打印挑选的变量
        self.clf.fit(x_train[md_var],y_train)                                                                   # 训练模型
    
        #------------模型评估与模型系数提取----------                                                        
        train_auc,train_ks = self._performance(self.clf,x_train[md_var],y_train,target='all')                   # 训练样本的KS,AUC
        test_auc,test_ks   = self._performance(self.clf,x_test[md_var],y_test,target='all')                     # 测试样本的KS,AUC
        w_norm = self.clf.coef_[0]                                                                              # 模型系数(对应归一化数据)
        b_norm = self.clf.intercept_[0]                                                                         # 模型阈值(对应归一化数据)
        w      = w_norm/(x_max[md_var]-x_min[md_var])                                                           # 模型系数(对应原始数据)
        b      = b_norm - (w_norm/(x_max[md_var] - x_min[md_var])).dot(x_min[md_var])                           # 模型阈值(对应原始数据)
        param_dict  = {'w':dict(zip(md_var, w)),'b':b}                                                          # 变量的参数字典
        
        self.var        = md_var                                                                                # 入模变量
        self.w          = np.array(w)                                                                           # 模型权重
        self.b          = np.array(b)                                                                           # 模型阈值
        self.w_norm     = w_norm                                                                                # 归一化训练得到的权重
        self.b_norm     = b_norm                                                                                # 归一化训练得到的阈值
        self.param_dict = param_dict                                                                            # 模型参数
        self.train_auc  = train_auc                                                                             # 训练样本的AUC
        self.train_ks   = train_ks                                                                              # 训练样本的KS
        self.test_auc   = test_auc                                                                              # 测试样本的AUC
        self.test_ks    = test_ks                                                                               # 测试样本的KS
        self._data_transfer._keep(self.var)                                                                     # 数据转换只保留有效变量
        print('\n建模结束........!\n')                                                                          # 打印进度
        return self                                                                                             # 返回结果
		
	# 预测函数	
    def predict(self,x):
        x = self._data_transfer.to_woe(x)                                                                       # 将数据转换为woe
        decision = x@self.w+self.b                                                                              # 计算判别值
        p = 1./(1+np.exp(-decision))                                                                            # 计算预测概率
        return p,decision                                                                                       # 返回概率与判别值
    

'''评分卡构建函数'''
def scoreCard(x,y,bin_sets,train_param={}):
	# bin_set的预处理
    if isinstance(bin_sets, str):                                                                               # 如果bin_sets是字符串
        if (bin_sets == 'grp'):                                                                                 # 如果bin_seta是'grp'
            bin_sets = {}                                                                                       # 初始化分箱统计
            for i in range(x.shape[1]):                                                                         # 逐个变量自动分箱
                var   = x.columns[i]                                                                            # 当前变量的名称
                cur_x = x.iloc[:,i]                                                                             # 变量的数据
                bset  = bins.merge.allEnum(cur_x)                                                               # 对当前变量分箱
                bin_sets[var] = bset                                                                            # 记录分箱结果
				
	# 逻辑回归模型训练参数预处理			
    test_size    = train_param['test_size']    if 'test_size'    in train_param else 0.2                        # 测试数据的比例
    penalty      = train_param['penalty']      if 'penalty'      in train_param else 'l2'                       # 是否正则化
    select_var   = train_param['select_var']   if 'select_var'   in train_param else True                       # 是否进行变量选择
    select_tol   = train_param['select_tol']   if 'select_tol'   in train_param else 0.005                      # 变量选择时使用的停止阈值
    random_state = train_param['random_state'] if 'random_state' in train_param else None                       # 随机种子
																								                
    # 构建评分卡                                                                                                
    data_transfer = _DataTransfer(x,y,bin_sets)                                                                 # 将数据转换为分箱    
    logit_model  = _LogitModel(data_transfer
                               ,test_size    = test_size                                                            
                               ,penalty      = penalty                                                               
                               ,select_var   = select_var                                                         
                               ,select_tol   = select_tol                                                         
                               ,random_state = random_state                                                     
                               )._fit(x,y)                                                                      # 将woe数据进行建模
    card = _Card(data_transfer,logit_model.param_dict).build()                                                  # 转评分
    return logit_model,card                                                                                     # 返回逻辑回归模型与评分卡
