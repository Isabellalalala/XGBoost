#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :class-Top 100.py
# @Time      :2023/7/7 15:37
# @Author    :Isabella 
import pandas as pd
from sklearn.model_selection import train_test_split

# 读取表达矩阵数据
expression_matrix = pd.read_csv('selected_genes_matrix.csv')

# 提取分类标签和样本名
labels = expression_matrix['class']
samples = expression_matrix['ent_name']

# 提取基因名
genes = expression_matrix.columns[2:103]

# 提取基因表达数据
gene_data = expression_matrix.iloc[:, 2:103]

# 分割数据集
samples_train, temp_samples = train_test_split(samples, test_size=0.4, random_state=42)
samples_valid, samples_test = train_test_split(temp_samples, test_size=0.5, random_state=42)

# 构建训练集
train_data = pd.concat([labels, samples, gene_data], axis=1)
train_data = train_data[train_data['ent_name'].isin(samples_train)]

# 保存为CSV文件
train_data.to_csv('train_top100.csv', index=False)

# 构建验证集
valid_data = pd.concat([labels, samples, gene_data], axis=1)
valid_data = valid_data[valid_data['ent_name'].isin(samples_valid)]

# 保存为CSV文件
valid_data.to_csv('valid_top100.csv', index=False)

# 构建测试集
test_data = pd.concat([labels, samples, gene_data], axis=1)
test_data = test_data[test_data['ent_name'].isin(samples_test)]

# 保存为CSV文件
test_data.to_csv('test_top100.csv', index=False)
