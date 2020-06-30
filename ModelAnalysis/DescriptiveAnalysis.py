# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 16:41:26 2020

@author: Mohd Azeem
"""

import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
import streamlit as st
from scipy import stats
from scipy.stats import norm,skew
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score,roc_auc_score,cohen_kappa_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from sklearn.ensemble import VotingClassifier

le = LabelEncoder()
sc = StandardScaler()
smote = SMOTETomek(random_state=42)
smote_over_sample = SMOTE('minority')

class DescriptiveAnalysis():
    
    def __init__(self,data,target):
        
        if data is not None:
            self.data = data
        
        self.target = target
        #self.columns = self.data.columns.to_list()
        self.skewed_columns = None
        self.res = None
        self.chi_list = list()
        self.X_train = pd.DataFrame()
        self.X_test = pd.DataFrame()
        self.y_train = pd.DataFrame()
        self.y_test = pd.DataFrame()
        self.features = pd.DataFrame()
        self.labels = pd.DataFrame()
        self.drop_columns_chi_list = list()
        self.classifiers = list()
        self.name = str()
        self.accuracy = float()
        self.roc_score = float()
        self.f1_score = float()
        self.precision = float()
        self.recall = float()
        self.kappa = float()
        self.model = dict()
        
        
        
    def check_for_Nan_values(self,choice='FALSE'):
        null_values_columns = [x for x in self.data.isnull().sum()]
        
        self.res = {}
        
        for key in list(self.data.columns):
            for value in null_values_columns:
                self.res[key] = value
                null_values_columns.remove(value)
                break
            
        if choice == 'TRUE':
            self.remove_Nan_values()
            
        else:
            return self.res
    
    
    
    def remove_Nan_values(self):
        remove_columns_nulls = [key for key,value in self.res.items() if np.float(value/self.data.shape[0]) >= 0.4]
        
        self.data.drop(remove_columns_nulls,axis=1,inplace=True)
        
        self.fill_Nan_values()
        
    def fill_Nan_values(self):
        numeric_features_null = [i for i in self.data.columns if self.data[i].isnull().sum() > 0 and self.data[i].dtypes != 'object']
        cat_features_null = [i for i in self.data.columns if self.data[i].isnull().sum() > 0 and self.data[i].dtypes == 'object']
        
        for i in numeric_features_null:
            self.data[i] = self.data[i].fillna(method='ffill')
            
        for i in cat_features_null:
            self.data[i] = self.data[i].fillna(self.data[i].mode()[0])
            
        self.data.dropna(inplace=True)     
            
        return self.res    
    
    
    def check_Nan_removed_or_not(self):
        
        if self.data.isnull().sum() == 0:
            return True
        else:
            return False
        
    
    def numerical_columns(self):
        numeric_list = list()
        
        for i in list(self.data.columns):
            if self.data[i].dtypes != 'object':
                numeric_list.append(i)
                
        return numeric_list
    
    def categorical_columns(self):
        categorical_list = list()
        
        for i in list(self.data.columns):
            if self.data[i].dtypes == 'object':
                categorical_list.append(i)
                
        return categorical_list
    
    def variables_columns(self):
        variable_list = list()
        variable_list = list(self.data.columns)
        variable_list.remove(self.target)
        
        return variable_list
    
    def Name_of_Target_Variable(self):
        
        return self.target
    
    def data_shape(self):
        
        return self.data.shape
    
    def length_of_dataframe(self):
        
        return self.data.shape[0]
    
    def binary_columns(self):
        
        binary_list = list()
        
        for i in list(self.data.columns):
            if len(self.data[i].unique()) == 2:
                binary_list.append(i)
                
        return binary_list

    def Ternary_columns(self):
         
        Ternary_list = list()
         
        for i in list(self.data.columns):
            if len(self.data[i].unique()) == 3:
                Ternary_list.append(i)
        
        return Ternary_list
    
    def describe_data(self):
        
        try:
            vars = ['Number of null values','number of numerical columns','number of categorical columns','Target Column','shape of the dataframe','size of dataframe','binary columns',
                    'ternary columns']
        
            cols = ['Description','Count']
            
            df = pd.DataFrame(columns=cols)
        
            df.loc[0] = [vars[0],len(self.check_for_Nan_values())]
            df.loc[1] = [vars[1],len(self.numerical_columns())]
            df.loc[2] = [vars[2],len(self.categorical_columns())]
            df.loc[3] = [vars[3],self.target]
            df.loc[4] = [vars[4],self.data_shape()]
            df.loc[5] = [vars[5],self.length_of_dataframe()]
            df.loc[6] = [vars[6],len(self.binary_columns())]
            df.loc[7] = [vars[7],len(self.Ternary_columns())]
            
            return df
            
        except Exception as e:
            return e
            
    def describe_numerical(self):
        
        cols = ['Name','Count','Unique','Median','Mean','Standard Deviation','25th Quantile','50th Quantile','75th Quantile']
        
        try:
            df = pd.DataFrame(columns=cols)
            
            numeric_columns = [i for i in self.data.columns if self.data[i].dtypes != 'object']
            
            i = 0
            
            for j in numeric_columns:
                df.loc[i] = [j,self.data[j].shape[0],len(self.data[j].unique()),np.median(self.data[j].dropna()),np.mean(self.data[j].dropna()),np.std(self.data[j].dropna()),np.percentile(self.data[j].dropna(),25),np.percentile(self.data[j].dropna(),50),np.percentile(self.data[j].dropna(),75)]
                
                i = i + 1
                
            return df
        
        except Exception as e:
            print('The error is',e)
            
    
    def describe_categorical(self):
        
        cols = ['Name','Count','Unique','Mode']
        
        try:
            df = pd.DataFrame(columns=cols)
            
            categorical_columns = [i for i in self.data.columns if self.data[i].dtypes == 'object']
            
            i = 0
            
            for j in categorical_columns:
                df.loc[i] = [j,self.data[j].shape[0],len(self.data[j].unique()),self.data[j].mode()[0]]
                
                i = i + 1
                
            return df
        
        except Exception as e:
            print('The error is',e)
            
    def detect_skewness(self):
        numeric_columns = [i for i in self.data.columns if self.data[i].dtypes != 'object']
        self.skewed_columns = list()
        threshold = 1.0
        
        for i in numeric_columns:
            
            if self.data[i].skew() > threshold:
                self.skewed_columns.append(i)
            
        return self.skewed_columns
    
    def remove_skewness(self):
        if self.target in self.skewed_columns:
            self.skewed_columns.remove(self.target)
       
        
        for i in self.skewed_columns:
			
            
            temp = np.log1p(self.data[i])
            self.data[i] = temp
            
        return self.data
    
    def chi_square_test(self,drop='FALSE'):

        categorical_columns = [i for i in self.data.columns if self.data[i].dtypes == 'object']
        
        length = len(self.data.columns)
        
        if type(self.target) == 'object':
            categorical_columns.remove(self.target)
        else:
            pass
        
        for i in categorical_columns:
            dataset_table=pd.crosstab(self.data[i],self.data[self.target])
            stat, p, dof, expected = stats.chi2_contingency(dataset_table)
            
            prob = 0.95
            
            critical = stats.chi2.ppf(prob, dof)
            
            if abs(stat) >= critical:
	            pass
            else:
                if drop == 'TRUE':
                    self.drop_columns_chi_list.append(i)
                    self.data.drop(i,axis=1,inplace=True)
                    
        return list(self.data.columns),self.drop_columns_chi_list
        
    def make_features_labels(self):
	
	    self.data.dropna(inplace=True)
        
        self.features = self.data.drop(self.target,axis=1)
        self.labels = self.data[self.target]
        
        #if self.labels.dtypes == 'object':
            #self.labels = le.fit_transform(self.labels)
        #else:
            #pass
        
        return self.features,self.labels
    
             
  
        
    def categorical_encoding(self):
        
        self.labels = le.fit_transform(self.labels)
        
        categorical_cols = [i for i in self.features.columns if self.features[i].dtypes == 'object']
        
        
        for i in categorical_cols:
            if len(self.features[i].unique()) == 2 or len(self.features[i].unique()) > 10:
                
                self.features[i] = le.fit_transform(self.features[i])
                #convert_dict = {i: int}
                self.features[i].astype('int')
                #self.features = self.features.astype(convert_dict)
                
        self.features = pd.get_dummies(self.features)
        
        return self.features,self.labels
    
    
    
        
        
        
        
        
    def apply_smote(self,choice='YES'):
        if len(self.data[self.target].unique()) > 2:
            self.features,self.labels = smote_over_sample.fit_sample(self.features,self.labels)
            
        else:
            self.features,self.labels = smote.fit_sample(self.features,self.labels)
            
    def check_imbalance_after_smote(self):
        
        for arr in self.labels:
            zero_els = np.count_nonzero(arr==0)
        
        one_els = len(self.labels) - zero_els
        some_list = [zero_els,one_els]
        return some_list    
        
        
        
    def split_train_test(self):
        
        self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(self.features,self.labels,test_size=0.15,random_state=42,shuffle=True)
        
        self.y_train = pd.DataFrame(self.y_train)
        self.y_test = pd.DataFrame(self.y_test)
        self.X_train = pd.DataFrame(self.X_train)
        self.X_test = pd.DataFrame(self.X_test)
        
        return self.X_train,self.X_test,self.y_train,self.y_test
    
    def data_shape_train(self):
        
        return self.X_train.shape
    
    def length_of_dataframe_train(self):
        
        return self.X_train.shape[0]
    
    def data_shape_test(self):
        
        return self.X_test.shape
    
    def length_of_dataframe_test(self):
        
        return self.X_test.shape[0]
    
    def standardized_data(self):
        
        self.X_train = sc.fit_transform(self.X_train)
        self.X_test = sc.transform(self.X_test)
        
        return self.X_train,self.X_test
    
    def create_classifiers(self,choice='All'):
        
        if choice == 'All':
            
            self.classifiers = [['DecisionTree :',DecisionTreeClassifier()],
               ['RandomForest :',RandomForestClassifier()], 
               ['Naive Bayes :', GaussianNB()],
               ['KNeighbours :', KNeighborsClassifier()],
               ['SVM :', SVC()],
               ['Neural Network :', MLPClassifier()],
               ['LogisticRegression :', LogisticRegression()],
               ['ExtraTreesClassifier :', ExtraTreesClassifier()],
               ['AdaBoostClassifier :', AdaBoostClassifier()],
               ['GradientBoostingClassifier: ', GradientBoostingClassifier()],
               ['XGBoost :', XGBClassifier()],
               ['CatBoost :', CatBoostClassifier(logging_level='Silent')]]
            
        elif choice == 'DecisionTree':
            self.classifiers = [['DecisionTree :',DecisionTreeClassifier()]]
            
        elif choice == 'RandomForest':
            self.classifiers = [['RandomForest :',RandomForestClassifier()]]
            
        elif choice == 'Naive Bayes':
            self.classifiers = [['Naive Bayes :', GaussianNB()]]
            
        elif choice == 'KNeighbours':
            self.classifiers = [['KNeighbours :', KNeighborsClassifier()]]
        
        elif choice == 'SVM':
            self.classifiers = [['SVM :',SVC()]]
            
        elif choice == 'Neural Network(MLP Classifier)':
            self.classifiers = [['Neural Network :', MLPClassifier()]]
        
        elif choice == 'LogisticRegression':
            self.classifiers = [['LogisticRegression :', LogisticRegression()]]
            
        elif choice == 'ExtraTreesClassifier':
            self.classifiers = [['ExtraTreesClassifier :', ExtraTreesClassifier()]]
            
        elif choice == 'AdaBoostClassifier':
            self.classifiers = [['AdaBoostClassifier :', AdaBoostClassifier()]]
            
        elif choice == 'GradientBoostingClassifier':
            self.classifiers = [['GradientBoostingClassifier: ', GradientBoostingClassifier()]]
        
        elif choice == 'XGBoost':
            self.classifiers = [['XGBoost :', XGBClassifier()]]
            
        elif choice == 'CatBoost':
            self.classifiers = [['CatBoost :', CatBoostClassifier(logging_level='Silent')]]
            
        else:
            pass
            
        
        
    def train_model(self):
        
        cols = ['Model Name','Accuracy','ROC_AUC Score','Precision','Recall','F1_Score',"Cohen's Kappa"]
        
        final_df = pd.DataFrame(columns=cols)
        i = 0
        
        for self.name,classifier in self.classifiers:
            classifier = classifier
            classifier.fit(self.X_train,self.y_train)
            predication = classifier.predict(self.X_test)
            self.accuracy = accuracy_score(self.y_test,predication)
            self.roc_score = roc_auc_score(self.y_test,predication)
            self.precision = precision_score(self.y_test,predication)
            self.recall = recall_score(self.y_test,predication)
            self.f1_score = f1_score(self.y_test,predication)
            self.kappa = cohen_kappa_score(self.y_test,predication)
            self.model[self.name] = {'Accuracy:':self.accuracy,'roc_auc_score':self.roc_score,'precision':self.precision,'recall':self.recall,'f1_score':self.f1_score,'cohen_kappa':self.kappa}
            final_df.loc[i] = [self.name,self.accuracy,self.roc_score,self.precision,self.recall,self.f1_score,self.kappa]
            
            i = i + 1
            
        return final_df
        
        
        
        
        
        
