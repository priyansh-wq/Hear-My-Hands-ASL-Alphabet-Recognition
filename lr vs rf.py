import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,LabelEncoder,PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv(r"C:\Users\sdivy\OneDrive\Documents\VS CODE\SML Project\asl_final_dataset.csv")

counts = df['label'].value_counts()
valid_classes = counts[counts >= 2].index
df = df[df['label'].isin(valid_classes)]

cat_col = df.select_dtypes(include=['object'])
le = LabelEncoder()
for col in cat_col:
    df[col]=le.fit_transform(df[col])
print("Label Encoding done")

x = df.drop('label',axis=1).values
y = df['label'].values

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
print("Standard Scaling done")

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(x_train_scaled,y_train)
print('LogisticRegression training done')

lr_pred = lr.predict(x_test_scaled)
acc_lr = accuracy_score(y_test,lr_pred)
print(f'Logistic Regression Accuracy: {acc_lr*100:.4f}%')


rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(x_train_scaled, y_train)
rf_pred = rf.predict(x_test_scaled)
rf_acc = accuracy_score(y_test, rf_pred)
print(f"Random Forest Accuracy: { rf_acc* 100:.4f}%")

cm = confusion_matrix(y_test, rf_pred)
plt.figure(figsize=(16, 14))
sns.heatmap(cm, annot=True, xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix - Random Forest')
plt.xlabel('Predicted Letter')
plt.ylabel('Actual Letter')
plt.show()