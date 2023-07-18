import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
# 读取训练数据
train_matrix = pd.read_csv('Data/origin/train/train_matrix.csv')
train_label = pd.read_csv('Data/origin/train/train_label.csv')

# 读取验证数据
valid_matrix = pd.read_csv('Data/origin/valid/valid_matrix.csv')
valid_label = pd.read_csv('Data/origin/valid/valid_label.csv')

# 读取测试数据
test_matrix = pd.read_csv('Data/origin/test/test_matrix.csv')
test_label = pd.read_csv('Data/origin/test/test_label.csv')

# 提取特征和标签
train_features = train_matrix.iloc[:, 1:].values
train_labels = train_label['class'].values

valid_features = valid_matrix.iloc[:, 1:].values
valid_labels = valid_label['class'].values

test_features = test_matrix.iloc[:, 1:].values
test_labels = test_label['class'].values

# 构建XGBoost分类器
model = xgb.XGBClassifier(objective='multi:softmax', num_class=20)

# 训练模型
model.fit(train_features, train_labels)

# 预测验证集
valid_pred_prob = model.predict_proba(valid_features)

# 计算准确率
valid_pred = model.predict(valid_features)
accuracy = accuracy_score(valid_labels, valid_pred)

# 计算AUC值
valid_auc = roc_auc_score(pd.get_dummies(valid_labels), valid_pred_prob, multi_class='ovr')

# 绘制ROC曲线
fpr, tpr, thresholds = roc_curve(pd.get_dummies(valid_labels).values.ravel(), valid_pred_prob.ravel())

# 输出准确率和AUC值
print('Validation Accuracy:', accuracy)
print('Validation AUC:', valid_auc)

# 绘制ROC曲线图
plt.plot(fpr, tpr, label='ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (Validation Set)')
plt.legend()
plt.show()

# 预测测试集
test_pred_prob = model.predict_proba(test_features)

# 计算AUC值
test_auc = roc_auc_score(pd.get_dummies(test_labels), test_pred_prob, multi_class='ovr')

# 绘制ROC曲线
fpr, tpr, thresholds = roc_curve(pd.get_dummies(test_labels).values.ravel(), test_pred_prob.ravel())

# 输出AUC值
print('Test AUC:', test_auc)

# 绘制ROC曲线图
plt.plot(fpr, tpr, label='ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (Test Set)')
plt.legend()
plt.show()
