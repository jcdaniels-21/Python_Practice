# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 14:54:54 2021

@author: Jarrod Daniels
"""
### Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import seaborn as sns
import sklearn
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.metrics import confusion_matrix


def main():
    df = readData()
    transformData(df)
    X, y = classifierPrep(df)
    X_train, X_test, y_train, y_test= ttsplit(X, y)
    model1 = buildModel(X_train, y_train)
    predictedValues = predict(model1, X_test)
    intercepts_coefficients(model1, y_test, predictedValues)
    #smLogit(X_train, y_train)
    confusionMatrix(y_test, predictedValues)


def readData():
    df = pd.read_csv('E:\Documents\GitHub\Python_Practice\Stroke Project\healthcare-dataset-stroke-data.csv')
    df = df.dropna()
    return df

def SmokerClass(CLASS):
        if CLASS == "formerly smoked":
            return 0
        elif CLASS == "never smoked":
            return 1
        elif CLASS == "smokes":
            return 2
        elif CLASS == "Unknown":
            return 3

def residenceClass(CLASS):
    if CLASS == "Urban":
        return 0
    elif CLASS == "Rural":
        return 1
    
def MarriedClass(CLASS):
    if CLASS == "No":
        return 0
    else:
        return 1

def workClass(CLASS):
    if CLASS == "Never_worked":
        return 0
    elif CLASS == "Self-employed":
        return 1
    elif CLASS == "Govt_job":
        return 2
    elif CLASS == "children":
        return 3
    elif CLASS == "Never_worked":
        return 4

def genderClass(CLASS):
    if CLASS == "Male":
        return 0
    elif CLASS == "Female":
        return 1
    elif CLASS == "Other":
        return 3
    
def transformData(df):
    df['smoking_status_num'] = df['smoking_status'].apply(SmokerClass)
    df['Residence_type_num'] = df['Residence_type'].apply(residenceClass)    
    df['gender_num'] = df['gender'].apply(SmokerClass)
    df['work_type_num'] = df['work_type'].apply(residenceClass)
    df['married_num'] = df['ever_married'].apply(SmokerClass)

def classifierPrep(df):
    X = df[['gender_num', 'age', 'hypertension', 'heart_disease', 'bmi', 'married_num', 'work_type_num', 'Residence_type_num', 'smoking_status_num']]
    y = df['stroke']
    print(list(X.columns.values)) 
    return X, y

def ttsplit(X, y):
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.20, random_state = 5)
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    return X_train, X_test, y_train, y_test

def buildModel(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

def predict(model, X_test):
    preds = model.predict(X_test)
    return preds


def intercepts_coefficients(model, y_test, predictions):
    print('Intercept: \n', model.intercept_)
    print('Coefficients: \n', model.coef_)
    for i in range(3):
        print('\n')
    print(classification_report(y_test,predictions))

    
def smLogit(X_train, y_train):
    logit_model=sm.MNLogit(y_train,sm.add_constant(X_train))
    logit_model
    result=logit_model.fit()
    stats1=result.summary()
    stats2=result.summary2()
    print(stats1)
    print(stats2)
    
def confusionMatrix(y_test, preds):
    print(confusion_matrix(y_test, preds))
    print('Accuracy Score:', metrics.accuracy_score(y_test, preds))  
    class_report=classification_report(y_test, preds)
    print(class_report)
    