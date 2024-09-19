#Churn Prediction

#Import important libraries
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_score, accuracy_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV

#Importing Data
df= pd.read_csv('Churn_Modelling.csv')

le = LabelEncoder()
df['Geography'] = le.fit_transform(df['Geography'])
df['Gender'] = le.fit_transform(df['Gender'])

##Remove RowNumber as its not an significant feature
df.drop(columns= ['RowNumber'], inplace= True)

##Lets put independent features together
features= ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']

X= df[features]
y= df['Exited']

##Splitting data for training and testing
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size= 0.2, random_state= 42)

###Feature Scaling
scaler= StandardScaler()
X_train= scaler.fit_transform(X_train)
X_test= scaler.transform(X_test)

###Lets apply Random Forest Classifier
model= RandomForestClassifier()
model.fit(X_train, y_train)

#Lets predict for testing
y_pred= model.predict(X_test)

##Model Evaluation: Cross Validation
from sklearn.model_selection import cross_val_score
cv_scores= cross_val_score(model, X, y, cv=5, scoring= 'accuracy')
print(cv_scores)

##Hyper Parameter Tuning:
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['auto', 'sqrt'],
    'bootstrap': [True, False]
}


grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='accuracy')
grid_search.fit(X_train, y_train)

print(grid_search.best_params_)
print(grid_search.best_estimator_)

best_model= grid_search.best_estimator_
y_pred_best= best_model.predict(X_test)

import streamlit as st 

Exited = {0: 'No Churn', 1: 'Churn'}

st.sidebar.title("Input Features")
CreditScore= st.sidebar.slider("CreditScore", float(df["CreditScore"].min()), float(df["CreditScore"].max()))
Geography= st.sidebar.slider("Geography", float(df["Geography"].min()), float(df["Geography"].max()))
Gender= st.sidebar.slider("Gender", float(df["Gender"].min()), float(df["Gender"].max()))
Age= st.sidebar.slider("Age", float(df["Age"].min()), float(df["Age"].max()))
Tenure= st.sidebar.slider("Tenure", float(df["Tenure"].min()), float(df["Tenure"].max()))
Balance= st.sidebar.slider("Balance", float(df["Balance"].min()), float(df["Balance"].max()))
NumOfProducts= st.sidebar.slider("NumOfProducts", float(df["NumOfProducts"].min()), float(df["NumOfProducts"].max()))
HasCrCard= st.sidebar.slider("HasCrCard", float(df["HasCrCard"].min()), float(df["HasCrCard"].max()))
IsActiveMember= st.sidebar.slider("IsActiveMember", float(df["IsActiveMember"].min()), float(df["IsActiveMember"].max()))
EstimatedSalary= st.sidebar.slider("EstimatedSalary", float(df["EstimatedSalary"].min()), float(df["EstimatedSalary"].max()))


input_data= [[CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary]]

prediction= best_model.predict(input_data)
status= Exited[prediction[0]]

st.write(prediction)
st.write(f"The churn prediction is {status}")