import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb 

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay 
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('C:/Users/Owner/Documents/reaction_data.csv')

print(df.head())
print(df.shape)
print(df.info())
print(df.describe())

features = ['Temperature' , 'pH' , 'Concentration' , 'Pressure']

# Check unique values in each column and handle non-numeric data
for col in features:
    print(f"Unique values in {col}: {df[col].unique()}")
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Handle missing values
print(df[features].isnull().sum())
df.dropna(subset=features, inplace=True)

# Visualize distributions
plt.figure(figsize=(12,8))
for i, col in enumerate(features):
    plt.subplot(2, 2, i+1)
    sb.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.tight_layout()
plt.show()

x = df[features]
y = df['Reaction_Success']
print(x.shape, y.shape) 

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

lr = LogisticRegression()
lr.fit(x_train, y_train)

svc = SVC()
svc.fit(x_train, y_train)

xgb = XGBClassifier()
xgb.fit(x_train, y_train)

from sklearn.metrics import accuracy_score 

models = {'Logistic Regression': lr, 'SVC': svc, 'XGBoost': xgb}

from sklearn.metrics import confusion_matrix

for name, model in models.items():
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy:.2f}")

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Failure", "Success"])
    fig, ax = plt.subplots(figsize=(5,5))
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    plt.title(f"{name} Confusion Matrix")
    plt.show()

