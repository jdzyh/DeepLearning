# coding=utf-8
import numpy as np
import pandas as pd

admissions = pd.read_csv('binary.csv')

# Make dummy variables for rank
# pd.concat函数连接多个数据
# pd.get_dummies函数根据字段生成rank_0,rank_1,...,rank_n; rank原来是k,则rank_k字段=1,其他字段=0
data = pd.concat([admissions, pd.get_dummies(admissions['rank'], prefix='rank')], axis=1)
data = data.drop('rank', axis=1)

# Standarize features
# 直接按照列进行计算标准化，并都改原有的值
for field in ['gre', 'gpa']:
    mean, std = data[field].mean(), data[field].std()
    data.loc[:,field] = (data[field]-mean)/std
    
# Split off random 10% of the data for testing
np.random.seed(42)
sample = np.random.choice(data.index, size=int(len(data)*0.9), replace=False)
data, test_data = data.ix[sample], data.drop(sample)

# Split into features and targets
#用于训练的数据集
features, targets = data.drop('admit', axis=1), data['admit']
# 获取训练完成后的测试数据集
features_test, targets_test = test_data.drop('admit', axis=1), test_data['admit']