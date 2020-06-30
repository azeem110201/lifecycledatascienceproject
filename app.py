# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 16:38:27 2020

@author: Mohd Azeem
"""


import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from ModelAnalysis.DescriptiveAnalysis import DescriptiveAnalysis
from EDA.EDAT import ExploratoryAnalysis
import base64


def main(df):
    
    #st.title('Data Cleaning, Exploratory Data Analysis, and Feature Selection Tool')
    
    #st.info(''' Welcome to our API where you can clean your data,explore the insights of your data using both descriptive and exploratory data analysis and mark out the features which are less important to the model for training ''')
    
    #st.warning('Note:This API works well with classification data (Structured Data which is in the form of tables (csv))')
    
    
    #def GetFile():
     #   uploaded_file = st.file_uploader("", type="csv")
      #  if uploaded_file is not None:
       #     return(pd.read_csv(uploaded_file))  

    #df = GetFile()
    
    
    
    target_name = st.selectbox(label='Select your target variable:',options=df.columns)    
    
    DA = DescriptiveAnalysis(df,target_name)
    
    st.sidebar.title('Descriptive Analysis')
    st.sidebar.subheader('Menu')
    
    if st.sidebar.checkbox('Click here for descriptive analysis'):
        st.title('Dataframe Basic Info')
        st.write('Data exploration is one of the key step and the first step in solving any machine learning promblem. We need to explore what type of variables(Numeric or categorical) and many more')
        
        if st.sidebar.checkbox('Head'):
            st.subheader('Dataframe Head:')
            st.write(df.head(10))
        
        if st.sidebar.checkbox('Data Description'):
            st.subheader('Data Description:')
            st.write(DA.describe_data())
            
        if st.sidebar.checkbox('Describe numerical features'):
            st.subheader('Numerical features:')
            st.write(DA.describe_numerical())

            
        if st.sidebar.checkbox('Describe categorical features'):
            st.subheader('categorical features:')
            st.write(DA.describe_categorical())
        
            
        if st.sidebar.checkbox('Names of Numerical Columns'):
            st.subheader('Numerical Column Names:')
            st.write(DA.numerical_columns())
            
        if st.sidebar.checkbox('Names of Categorical Columns'):
            st.subheader('Categorical Column Names:')
            st.write(DA.categorical_columns()) 
            
        if st.sidebar.checkbox('Target Variable is'):
            st.subheader('Target Variable:')
            st.write(DA.Name_of_Target_Variable()) 
            
        if st.sidebar.checkbox('Shape of the data'):
            st.subheader('Data shape:')
            st.write(DA.data_shape())
            
        if st.sidebar.checkbox('Number of rows in data'):
            st.subheader('length of the data:')
            st.write(DA.length_of_dataframe())
            
        if st.sidebar.checkbox('Binary Columns in data'):
            st.subheader('Binary columns:')
            st.write('Binary columns are those features which have binary or two unique elements in their features.Ex:Male and Female')
            st.write(DA.binary_columns())
            
        if st.sidebar.checkbox('Ternary Columns in data'):
            st.subheader('Ternary columns:')
            st.write('Ternary columns are those features which have three unique elements in their features.Ex:Low, Medium and high')
            st.write(DA.Ternary_columns())    
            

    
    st.sidebar.title('Data Cleaning')
    st.sidebar.subheader('Menu')
    
    if st.sidebar.checkbox('Click here for Data Cleaning options'):
        st.title('Check for Nan values')
        
        if st.sidebar.checkbox('Check Nan values and remove the column which having more than 40% of Nan values'):
            st.subheader('Nan values')
            
            st.write(df.isnull().sum())
            
            choice = st.selectbox(label='Check for Nan values and remove the column which having more than 40% of Nan values',options=['TRUE','FALSE'])
            
            
            st.write(DA.check_for_Nan_values(choice))
            
            st.write(df.isnull().sum())
            
    EA = ExploratoryAnalysis(df,target_name)        
            
    st.sidebar.title('Data Preprocessing')
    st.sidebar.subheader('Menu')
    
    if st.sidebar.checkbox('Click here for Data Preprocessing options'):
        st.title('Check How your data is distributed')
        
        if st.sidebar.checkbox('Skewness of numerical columns'):
            st.subheader('Skewness')
            st.write('The numerical columns that are skewed:')
            st.write(DA.detect_skewness())
            
            choice = st.selectbox(label='Select the columns from above skewed columns to see how they are distributed',options=DA.detect_skewness())
            
            fig = EA.distplot(choice)
            st.pyplot()
            
        if st.sidebar.checkbox('Click if you want to remove skewness from the data'):
            st.subheader('Remove skewness using log(x+1) as transformation factor')
            st.write('DataFrame before removing skewness')
            st.write(df.head(10))
            st.write('DataFrame after removing skewness')
            st.write(DA.remove_skewness())
            st.info('Note the numerical values of the above two DataFrames')
            
    ##graphs
    
    
    
    
    
    st.sidebar.title('Feature engineering')
    st.sidebar.subheader('Menu')
    
    if st.sidebar.checkbox('Click here for Feature Engineering options'):
        st.title('Feature Enginnering')
        st.write('Feature engineering is one of the most important step in the process of solving machine learning promblems. Here you omit the features that are least important w.r.t target variable and this helps in reducing the dimensions of the dataframe and select the only the important features for the model')
        
        if st.sidebar.checkbox('Chi-Square Test'):
            st.subheader('Chi-Square Test for features w.r.t target variable')
            st.write('columns before Chi-Square test:')
            st.write(df.columns)
            choice = st.selectbox(label='Do you want to drop the columns which does not satisfy chi-Square test?',options=['TRUE','FALSE'])
            st.write('columns after Chi-Square test:')
            chi_list_column,droped_columns_chi_list = DA.chi_square_test(choice)
            st.write(chi_list_column)
            st.write('Columns that are dropped in chi-square test')
            st.write(droped_columns_chi_list)
            
            
        if st.sidebar.checkbox('Make features and Labels'):
            st.subheader('Make Features and Labels for training the data')
            features,labels = DA.make_features_labels()
            st.write('features:')
            st.write(features.head(10))
            st.write('labels:')
            st.write(labels[:10])
            
        if st.sidebar.checkbox('Categorical Encoding'):
            st.subheader('Encode the columns which have data-type as string. Here the columns which have less than 3 unique values and more than 10 unique values in them are encoded with LabelEncoder and the others are encoded with One-Hot Encoding. This is the most preferred way of encoding the categorical columns')
            features,labels = DA.categorical_encoding()
            st.write('features:')
            st.write(features.head(10))
            st.write('labels:')
            st.write(labels[:10])
        
        
        
        if st.sidebar.checkbox('Check for imbalance labels in the dataset'):
            st.subheader('Check the for imbalanced label in dataet')
            st.warning('Check If the labels are balanced or not. Ignoring the datasets which are Imbalance can lead to serious promblems')
            #choice = st.selectbox(label='Please see below the count of each value in the output labels. If there count is almost the same than leave if and if not then please click on yes if you want to balance the data.',options=['YES','NO'])
            st.write(df[target_name].value_counts())
            st.write(DA.apply_smote())
            ##changes were made from here
            #some_list = st.write(DA.check_imbalance_after_smote())
            #df = df.dropna(inplace=True)
            #fig = EA.countplot(df,some_list)
            #st.pyplot()
            
            
        if st.sidebar.checkbox('Train Test Split'):
            st.subheader('Split the dataset into train and test')
            X_train,X_test,y_train,y_test = DA.split_train_test()
            
            st.write('This is our training set')
            st.write(X_train.head(10))
            st.write('This is our testing set')
            st.write(X_test.head(10))
            
            st.write('Shape of training set')
            st.write(DA.data_shape_train())
            st.write('Size of training set')
            st.write(DA.length_of_dataframe_train())
            
            st.write('Shape of testing set')
            st.write(DA.data_shape_test())
            st.write('Size of testing set')
            st.write(DA.length_of_dataframe_test())
            
        if st.sidebar.checkbox('Standardize data'):
            st.subheader('standardization of data')
            st.write('Always standardize the data so that all the values are as closer to each other as possible (Here the data will be Normally distributed i.e mean = 0 and standar devaition = 1)')
            features,labels = DA.standardized_data()
            
            st.write('The data has been standarised')
            
            st.write('The next step is training of the model. If you are providing very large datasets it may take time to evaluate and give you the results')
    

    #graphs        
    st.sidebar.title('Graphs')
    st.sidebar.subheader('Menu')
    
    if st.sidebar.checkbox('Click here to see some graphical or visualization representation of the dataframe'):
        st.title('Graphs')
        
        
        if st.sidebar.checkbox('Count Plot'):
            
            st.subheader('Count Plot')
            choice = st.selectbox(label='Select the column for CountPlot:',options=df.columns)
            choice_hue = st.selectbox(label='select hue: (Optional)',options=df.columns)
            
            fig = EA.countplot(choice, choice_hue)
            st.pyplot()
            
            
        if st.sidebar.checkbox('Distribution Plot'):
            
            st.subheader('Distribution Plot')
            choice = st.selectbox(label='Select the column for distribution Plot:',options=df.columns)
    
            fig = EA.distplot(choice)
            st.pyplot()
            
        
        if st.sidebar.checkbox('kdeplot'):
            
            st.subheader('kdeplot')
            choice_1 = st.selectbox(label='Select the first column for kdeplot:',options=df.columns)
            choice_2 = st.selectbox(label='Select the second column for kdeplot:',options=df.columns)
            shade = st.selectbox(label='Do you want to have shade for your kdeplot (Optional)',options=[None,True])
    
            fig = EA.kdeplot(choice_1,choice_2,shade)
            st.pyplot()
            
            
            
        if st.sidebar.checkbox('Joint plot'):
            
            st.subheader('Joint Plot')
            choice_1 = st.selectbox(label='Select the first column for joint plot:',options=df.columns)
            choice_2 = st.selectbox(label='Select the second column for joint plot:',options=df.columns)
    
            fig = EA.joint_plot(choice_1,choice_2)
            st.pyplot()
            
            
        if st.sidebar.checkbox('Point plot'):
            
            st.subheader('Point plot')
            choice_1 = st.selectbox(label='Select the first column for point plot:',options=df.columns)
            choice_2 = st.selectbox(label='Select the second column for point plot:',options=df.columns)
    
            fig = EA.pointplot(choice_1,choice_2)
            st.pyplot()
            
            
            
        if st.sidebar.checkbox('Line plot'):
            
            st.subheader('Line plot')
            choice_1 = st.selectbox(label='Select the first column for Line plot:',options=df.columns)
            choice_2 = st.selectbox(label='Select the second column for Line plot:',options=df.columns)
            choice_hue = st.selectbox(label='Select hue: (Optional)',options=df.columns)
    
            fig = EA.pointplot(choice_1,choice_2,choice_hue)
            st.pyplot()
            
            
        if st.sidebar.checkbox('boxplot'):
            
            st.subheader('Box plot')
            choice_1 = st.selectbox(label='Select the first column for Box plot:',options=df.columns)
            choice_2 = st.selectbox(label='Select the second column for Box plot:',options=df.columns)
            choice_hue = st.selectbox(label='Select a hue : (Optional)',options=df.columns)
    
            fig = EA.boxplot(choice_1,choice_2,choice_hue)
            st.pyplot()
            
            
            
        if st.sidebar.checkbox('Heatmap'):
            
            st.subheader('Heatmap')
            
            choice_annot = st.selectbox(label='Select whether to annot the heatmap or not: (Optional)',options=[None,True])
    
            fig = EA.heatmap(choice_annot)
            st.pyplot()        
    

    
    st.sidebar.title('Training and Predication')
    st.sidebar.subheader('Menu')    
    if st.sidebar.checkbox('Click here for Training and Predication options'):
        
        def get_table_download_link(df):
            """Generates a link allowing the data in a given panda dataframe to be downloaded
            in:  dataframe
            out: href string
            """
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(
                csv.encode()
            ).decode()  # some strings <-> bytes conversions necessary here
            
            return f'<a href="data:file/csv;base64,{b64}" download="mypredications.csv">Download your predication table in the form of csv</a>'
        
        
        
        
        
        st.title('Training and Predication of your model')
        st.info('Here your model will get trained with different Machine Learning algorithms')
        st.warning('Below if you want to train with "ALL" the different machine learning algorithms and if your dataset is large than it may take time to get your results')
        choice = st.selectbox('Which Algorithm you want to choose (All means it will return the result for all the different machine learning algorithms used)',options=['All','DecisionTree','RandomForest','Naive Bayes','KNeighbours','SVM','Neural Network(MLP Classifier)','LogisticRegression','ExtraTreesClassifier','AdaBoostClassifier','GradientBoostingClassifer','XGBoost','CatBoost'])
         
        DA.create_classifiers(choice)
        models = DA.train_model()
        st.write(models)
         
         
        
        st.markdown(get_table_download_link(models), unsafe_allow_html=True)
         
        
        
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
        
            
            
            
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
if __name__ == '__main__':
    
    st.title('AutoML:Train your models by just providing your dataset and target feature')
    
    st.info(''' Welcome to our API where you can clean your data,explore the insights of your data using both descriptive and exploratory data analysis and mark out the features which are less important to the model for training ''')
    
    st.warning('Note:This API works well with classification data (Structured Data which is in the form of tables (csv))')
    
    def GetFile():
        uploaded_file = st.file_uploader("", type="csv")
        if uploaded_file is not None:
            return(pd.read_csv(uploaded_file,sep=,))  

    df = GetFile()
    
    if df is not None:
        main(df)
    
    #main(df)
    
    
    
