#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :test.py
# @Time      :2023/5/30 11:21
# @Author    :Isabella 
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder

# 读取训练数据
train_matrix = pd.read_csv('Data/origin/train/train_matrix.csv')
train_label = pd.read_csv('Data/origin/train/train_label.csv')

# 读取验证数据
valid_matrix = pd.read_csv('Data/origin/valid/valid_matrix.csv')
valid_label = pd.read_csv('Data/origin/valid/valid_label.csv')

# 读取测试数据
test_matrix = pd.read_csv('Data/origin/test/test_matrix.csv')
test_label = pd.read_csv('Data/origin/test/test_label.csv')

# 获取样本名称列名和基因列名
sample_col = 'ent_name'
gene_cols = train_matrix.columns[1:]

# 对分类标签进行编码
label_encoder = LabelEncoder()
train_label_encoded = label_encoder.fit_transform(train_label['class'])
valid_label_encoded = label_encoder.transform(valid_label['class'])
test_label_encoded = label_encoder.transform(test_label['class'])

# 构建DMatrix数据集
dtrain = xgb.DMatrix(train_matrix[gene_cols], label=train_label_encoded)
dvalid = xgb.DMatrix(valid_matrix[gene_cols], label=valid_label_encoded)
dtest = xgb.DMatrix(test_matrix[gene_cols], label=test_label_encoded)


# 训练模型
model = xgb.train(params, dtrain, num_boost_round=1000, early_stopping_rounds=10, evals=[(dvalid, 'valid')])

# 在验证集上预测
valid_preds = model.predict(dvalid)

# 在测试集上预测
test_preds = model.predict(dtest)
# 设置XGBoost的参数
params = {
    'objective': 'multi:softprob',
    'num_class': 20,
    'eval_metric': ['mlogloss', 'merror'],
    'eta': 0.1,
    'max_depth': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42
}

# 计算AUC值
valid_auc = roc_auc_score(pd.get_dummies(valid_label_encoded), valid_preds, average='weighted', multi_class='ovr')
test_auc = roc_auc_score(pd.get_dummies(test_label_encoded), test_preds, average='weighted', multi_class='ovr')
print("Validation AUC: {:.4f}".format(valid_auc))
print("Test AUC: {:.4f}".format(test_auc))

# 绘制ROC曲线
valid_fpr, valid_tpr, _ = roc_curve(pd.get_dummies(valid_label_encoded).values.ravel(), valid_preds.ravel())
test_fpr, test_tpr, _ = roc_curve(pd.get_dummies(test_label_encoded).values.ravel(), test_preds.ravel())

# 可视化ROC曲线
import matplotlib.pyplot as plt

plt.figure()
plt.plot(valid_fpr, valid_tpr, label='Validation')
plt.plot(test_fpr, test_tpr, label='Test')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
