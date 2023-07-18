#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :test_Top 100.py
# @Time      :2023/7/11 10:31
# @Author    :Isabella
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from scipy.special import softmax

# 读取训练数据
train_data = pd.read_csv('Data/Top 100 Gene/train_top100.csv')
train_labels = train_data['class']
train_features = train_data.iloc[:, 2:]

# 读取验证数据
valid_data = pd.read_csv('Data/Top 100 Gene/valid_top100.csv')
valid_labels = valid_data['class']
valid_features = valid_data.iloc[:, 2:]

# 读取测试数据
test_data = pd.read_csv('Data/Top 100 Gene/test_top100.csv')
test_labels = test_data['class']
test_features = test_data.iloc[:, 2:]

# 将数据转换为DMatrix格式
dtrain = xgb.DMatrix(train_features, label=train_labels)
dvalid = xgb.DMatrix(valid_features, label=valid_labels)
dtest = xgb.DMatrix(test_features, label=test_labels)

# 设置XGBoost的参数
#'num_class': 20：指定类别的数量，代表共20种分类（0-19）。
params = {
    'objective': 'multi:softmax',
    'num_class': 20,
    'eval_metric': 'mlogloss'
}

# 训练模型
model = xgb.train(params, dtrain, num_boost_round=100, evals=[(dvalid, 'valid')], early_stopping_rounds=10)

# 在验证集上进行预测
valid_pred = model.predict(dvalid)
valid_pred_scores = model.predict(dvalid, output_margin=True)
valid_pred_proba = softmax(valid_pred_scores, axis=1)

# 计算准确率和loss值
valid_accuracy = accuracy_score(valid_labels, valid_pred)
valid_loss = model.best_score

# 在测试集上进行预测
test_pred = model.predict(dtest)
test_pred_scores = model.predict(dtest, output_margin=True)
test_pred_proba = softmax(test_pred_scores, axis=1)

# 计算AUC值
auc_score = roc_auc_score(pd.get_dummies(test_labels), test_pred_proba, multi_class='ovr')

# 输出准确率和AUC值
print('Validation Accuracy:', valid_accuracy)
print('Validation AUC:', auc_score )

# 计算ROC曲线
fpr, tpr, _ = roc_curve(pd.get_dummies(test_labels).values.ravel(), test_pred_proba.ravel())

# 可视化AUC和ROC曲线
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()
