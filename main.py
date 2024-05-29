import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# 載入數據集
data = pd.read_csv('student-por.csv')



# 數據預處理
# 編碼二元類別變量
binary_columns = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'schoolsup',
                  'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']
for column in binary_columns:
    data[column] = LabelEncoder().fit_transform(data[column])

# 編碼名義類別變量
nominal_columns = ['Mjob', 'Fjob', 'reason', 'guardian']
data = pd.get_dummies(data, columns=nominal_columns, drop_first=True)

# 選擇特徵和目標變量
features = ['Medu', 'Fedu', 'age', 'address', 'famsize', 'Pstatus', 'traveltime',
            'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities',
            'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout',
            'Dalc', 'Walc', 'health', 'absences']
X = data[features]
y = (data['G3'] >= 10).astype(int)  # 假設G3為最終成績，並將其二元化（通過或不通過）

# 將數據分為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 構建邏輯回歸模型
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 模型評估
# 交叉驗證
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
print("交叉驗證分數:", cv_scores)

# 測試集評估
y_pred = model.predict(X_test)
print("準確率:", accuracy_score(y_test, y_pred))
print("混淆矩陣:\n", confusion_matrix(y_test, y_pred))
print("分類報告:\n", classification_report(y_test, y_pred))

# 分析與解釋
# 解釋模型係數
coefficients = model.coef_
feature_names = X.columns
coef_df = pd.DataFrame(coefficients, columns=feature_names)
print("模型係數:\n", coef_df)
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.scatterplot(x='Medu', y='G3', data=data)
plt.title('mom education vs final grade')
plt.xlabel('mom education')
plt.ylabel('final grade')

plt.subplot(1, 2, 2)
sns.scatterplot(x='Fedu', y='G3', data=data)
plt.title('dad education vs final grade')
plt.xlabel('dad education')
plt.ylabel('final grade')

plt.tight_layout()
plt.show()

# 繪製箱線圖
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.boxplot(x='Medu', y='G3', data=data)
plt.title('mom education vs final grade (boxplot)')
plt.xlabel('mom education')
plt.ylabel('final grade')

plt.subplot(1, 2, 2)
sns.boxplot(x='Fedu', y='G3', data=data)
plt.title('dad education vs final grade (boxplot)')
plt.xlabel('dad education')
plt.ylabel('final grade')

plt.tight_layout()
plt.show()