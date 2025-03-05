# bbbrisk-bbb风控包
本代码由bbbdata为评分卡专门开发的小贷风控包，目前提供的功能主要是构建评分卡

bbbrisk的优点是，不仅仅是单纯的提供评分卡模型，而是实际构建评分卡时一切可能用到的功能(例如，变量的分析、分箱，模型的训练，评分卡表的构建，最终的报告等等)

评分卡教程、API说明、源代码详见文末



## Install and Upgrade · 安装与升级

```bash
pip install bbbrisk    # to install
pip install -U bbbrisk # to upgrade
```
bbbrisk依赖于numpy、pandas和sklearn包，最好尽可能保障已经安装了这三个包



## Key features · 主要功能
bbbrisk包括(不局限)如下功能：

- **自带小贷数据**

1. 小贷原始变量数据

2. 小贷分箱变量数据

- **对变量进行分箱与分** 

1. 打印单变量手动分箱的结果

2. 打印单变量自动分箱结果

   等频、等距、卡方分箱、KS分箱、决策树分箱等
   
3. 对所有变量直接自动分箱

- **数据转换** 

1. 将数据转换为分箱数据

2. 将分箱数据转换为woe数据

- **构建模型与评分卡** 

1. 支持直接用原始数据与分箱设置进行建模

2. 支持使用分箱后的数据进行建模

3. 支持设置模型转评分的参数

4. 支持按分数范围进行模型转评分

5. 支持模型预测(输出概率)

6. 支持评分预测(输出评分)

- **写报告时需要的内容** 

1. 计算阈值表

2. 绘画分数分布图

更多功能参见每个API的说明与示例~




## Demo · 使用示例

```python
import bbbrisk as br

#加载数据                                                              
data = br.datasets.load_bloan()                                        # 加载数据
x = data.iloc[:,:-1]                                                   # 变量数据
y = data['is_bad']                                                     # 标签数据

# 构建评分卡
bin_sets  = br.bins.batch.autoBins(x, y,enum_var=['city','marital'])   # 自动分箱,必须指出哪些是枚举变量
model,card = br.model.scoreCard(x,y,bin_sets)                          # 构建评分卡
score      = card.predict(x[card.var])                                 # 用评分卡进行评分

# 计算阈值表与分数分布图
thd_tb    = br.report.get_threshold_tb(score,y,bin_step=10)            # 阈值表
br.report.draw_score_disb(score,y,bin_step=10,figsize=(14, 4))         # 分数分布

# 打印结果
print('\n-----【 模型性能评估 】----')
print('* 模型训练AUC:',model.train_auc)                                # 打印模型训练数据集的AUC
print('* 模型测试AUC:',model.test_auc)                                 # 打印模型测试数据集的AUC
print('* 模型训练KS:',model.train_ks)                                  # 打印模型训练数据集的KS
print('* 模型测试KS:',model.test_ks)                                   # 打印模型测试数据集的KS
													                   
print('\n--------【 模型 】---------')                                 
print('* 模型使用的变量:',model.var)                                   # 模型最终使用的变量
print('* 模型权重:',model.w)                                           # 模型的变量权重
print('* 模型阈值:',model.b)                                           # 模型的阈值
													                   
print('\n--------【 评分卡 】---------')    
print('\n* 特征得分featureScore:   \n' ,card.featureScore      )       # 特征得分
print('\n* 基础得分baseScore:        ' ,card.baseScore         )       # 基础分
```
在上述代码中,展示了如何使用自动分箱构建一个评分卡，并打模型的AUC、KS，以及构建好的评分卡表，

进一步地，用构建好的评分卡对样本进行评分，并计算出阈值表与分数分布图




## Documents · 文档
- [评分卡教程与api说明](https://www.bbbdata.com/ml)

- [github](https://github.com/bbbdata/bbbrisk)

## 更新日志
- v1.0.2: 优化函数名称与函数功能
- v1.0.1: 添加数据
- v1.0.0: 初步上传