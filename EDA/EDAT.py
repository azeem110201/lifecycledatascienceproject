# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 20:04:13 2020

@author: Mohd Azeem
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from scipy.stats import norm,skew


class ExploratoryAnalysis():
    
    def __init__(self,data,target):
        self.data = data
        self.target = target
        self.columns = self.data.columns
        
    def countplot(self,column,hue=None):
        plt.figure(figsize=(15,8))
        return sns.countplot(data=self.data,x=column, hue=hue)
        
    def heatmap(self,annot=None,cmap="YlGnBu"):
        plt.figure(figsize=(15,8))
        matrix_corr = self.data.corr()
        mask = np.zeros_like(matrix_corr)
        mask[np.triu_indices_from(mask)] = True
        return sns.heatmap(matrix_corr, linewidths=.4,annot=annot,cmap=cmap,mask=mask)
    
    def distplot(self,column):
        plt.figure(figsize=(15,8))
        #plt.xticks(rotation = 90)
        return sns.distplot(self.data[column],fit=norm)
    
    
    def kdeplot(self,column_1,column_2=None,shade=None,cbar=None):
        plt.figure(figsize=(15,8))
        return sns.kdeplot(self.data[column_1],self.data[column_2],shade=shade,color='r',cbar=cbar)
    
    def joint_plot(self,column_1,column_2,kind='reg'):
        plt.figure(figsize=(15,8))
        return sns.jointplot(x=column_1,y=column_2,data=self.data,kind=kind)
    
    def boxplot(self,column_1,column_2=None,hue=None):
        plt.figure(figsize=(15,8))
        return sns.boxplot(x=column_1,y=column_2,data=self.data,hue=hue,palette="Set3")
    
    def pointplot(self,column_1,column_2,hue=None):
        plt.figure(figsize=(15,8))
        return sns.pointplot(x=column_1,y=column_2,data=self.data,hue=hue)
    
    def lineplot(self,column_1,column_2,hue=None):
        plt.figure(figsize=(15,8))
        return sns.lineplot(x=column_1,y=column_2,data=self.data,hue=hue)
    
    
    
    
    
    
    
   
