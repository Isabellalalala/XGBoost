#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :class.py
# @Time      :2023/6/2 14:38
# @Author    :Isabella 
import pandas as pd
from sklearn.model_selection import train_test_split

# 读取total.csv文件，跳过第一行作为列名
data = pd.read_csv('Data/total.csv', skiprows=1)

# 将数据分为训练集、验证集和测试集
train_data, temp_data = train_test_split(data, test_size=0.4, random_state=42)
valid_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# 保存分割后的数据为CSV文件
train_data.to_csv('train.csv', index=False)
valid_data.to_csv('valid.csv', index=False)
test_data.to_csv('test.csv', index=False)
