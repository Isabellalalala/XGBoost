#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :Selected-Top 100.py
# @Time      :2023/7/7 11:15
# @Author    :Isabella
import pandas as pd

# 读取基因列表
gene_list = pd.read_csv("Data/Top 100 Gene/Top 100 Genes.csv", header=None, skiprows=1)
gene_names = gene_list.iloc[:, 0].tolist()  # 提取基因名

# 读取特征矩阵
expression_matrix = pd.read_csv("Data/origin/expression_matrix.csv", index_col=0)

# 提取指定基因的特征矩阵
selected_genes_matrix = expression_matrix.loc[:, gene_names]

# 输出新的特征矩阵
selected_genes_matrix.to_csv("selected_genes_matrix.csv", index=False)
