# AutoML
Get the EDA Analysis,Descriptive Analysis predications for your binary classification model just my adding your dataset and giving the target(dependent) variable

## Table of Contents
*  Demo
* Overview
* Motivation
* Inspiration
* Technical Aspect
* Advantages and limitations of this application
* Installation
* Deployment using Heroku App
* Directory Tree
* To Do
* Credits

## Demo

Link: https://automlmodels.herokuapp.com/

## Overview
This is a simple application built using streamlit library. In this application you can explore about Descriptive Analysis, EDA analysis and Model evaluation on differnt metrics using different machine learning algorithm. Note that this application can only evalute your model if it is binary classification task and in upcoming updates I will try to expand this to Regression task and multi-class classification.

## Motivation
Every time when we sit to solve any promblem(classification or regression) on kaggle, we have to follow certain steps to get a better accuracy of the model. Here's an application which can hep you solve this im just fraction of minutes. This application has also led me understand how a end-to-end Machine learning process is done.

## Inspiration
I got the inspiration from a famous application called as Pycaret. Here you just have to give your dataset and target variable and it will evaluate your model on different evaluation metrics and give you the result. But there are some additional options that you get in my application such as EDA exploration, Descriptive analysis amd many more.

## Technical Aspects
The code in this application is wriiten in python and it was deployed using Heroku-app. There is an awesome library which I have used i.e streamlit which helps in creating front-end part of the application and helps in interacting with the code written at back-end. I have used some popular library such as sklearn,seaborn,matplotlib,pandas and numpy. Besides that I have also included some of the modern Machine Learning algorithms such as Xgboost and catboost. After the evaluation of the model, the results are displayed in the form of a dataframe which you can download in your local machine.

## Advantages and limitations of this application
The most important advantage of using this application is that it helps in reducing the load from a person who is from non-technical backgrounda and don't have indepth knowledge of machine learning algorithms. This application helps you to visualize the data and find out its descriptive properties. It helps in removing Nan values,skewness of the data and encodes the categorical variables into label Encoder and One-Hot representation. On the other hand the limitations for this application is that, it is bit slow when dealing with large datasets and it can only handle binary classification,but don't worry I will bring improvements to this application so that all the end user can access this.

## Installation
The Code is written in Python 3.7. If you don't have Python installed you can find it here. If you are using a lower version of Python you can upgrade using the pip package, ensuring you have the latest version of pip. To install the required packages and libraries, run this command in the project directory after cloning the repository " pip install -r requirements.txt "

# Deployment using heroku-app
This application is deployed on heroku.Heroku is a platform as a service (PaaS) that enables developers to build, run, and operate applications entirely in the cloud.

# Directory Tree

├── DescriptiveAnalysis 
│   ├── DescriptiveAnalysis.py
├── ExploratoryAnalysis
│   ├── ExploratoryAnalysis.py
├── app.py
├── Procfile
├── requirements.txt
├── Procfile
├── README.md
└── setup.sh

# To-Do list
In the upcoming updates, I will try to add many more options to the application like regression task, NLP task and I will try improve the existing application program for better user experience.

# Credits
* I want to thank Syed Abrar (Github link:https://github.com/Abrar-04) and Sufiyan Abdullah Ghori (Github link:https://github.com/MSufiyanAG) for always motivating me in the field of Machine Learning
* I want to thank Krisk Naik (Github link:https://github.com/krishnaik06) for always motivating us and helping us learn the concepts of Machine and Deep learning and also helping us familiarizing the tools and technology updates in this field.














