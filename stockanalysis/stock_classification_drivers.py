#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor 
from sklearn.metrics import mean_squared_error, r2_score 
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor

# Feature Scaling
from sklearn.preprocessing import StandardScaler

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

#import pandas_datareader.data as web
import datetime as dt

from stockdata import prepare_stockdata

from stock_classifiers import stock_classifiers


# In[15]:




# In[19]:


class classification_drivers:
    #def __init__(self):
    #    self.cls = stock_classifiers()
        
    def linearregression_driver(self,x_train,y_train,inp):
        cls = stock_classifiers()
        #Train the model
        lr = cls.linear_regression(x_train,y_train)
        #Evaluate the model performance
        cls.lr_score(lr,x_test,y_test)
        # Predict Next Day Value
        lr_pred = cls.model_prediction(lr,inp)
        #print("Linear Regression Predicted Next Day Value : ", lr_pred)
        return lr,lr_pred

    def randomforest_driver(self,x_train,x_test,inp):
        cls = stock_classifiers()
        # Feature scaling 
        sc,x_train_sc, x_test_sc = cls.feature_scaling(x_train,x_test)

        #Train the model
        rf = cls.random_forest(x_train_sc,y_train,40)

        #Evaluate the model performance
        cls.model_score(rf,x_test_sc,y_test)

        #Evaluate the model accuracy
        y_pred=rf.predict(x_test_sc)
        
        cls.model_error(y_test,y_pred)

        #Input scaling
        inp_sc = sc.transform(inp)

        # Prediction
        rf_pred = cls.model_prediction(rf,inp_sc)

        print("Predicted Next Day Value for Random Forest : ", rf_pred)

        return rf,rf_pred

    def decisiontree_driver(self,x_train,x_test,inp):    
        cls = stock_classifiers()
        #Train the model
        model = cls.decision_tree(x_train,y_train)
        #Evaluate the model performance
        cls.model_score(model,x_test,y_test)
        # Predict Next Day Value
        dt_pred = cls.model_prediction(model,inp)
        #print("Decision Tree Predicted Next Day Value : ", dt_pred)
        return model,dt_pred

    def lasso_driver(self, x_train,x_test,inp):    
        cls = stock_classifiers()
        #Train the model
        model = cls.lasso(x_train,y_train)
        #Evaluate the model performance
        cls.model_score(model,x_test,y_test)
        # Predict Next Day Value
        lasso_pred = cls.model_prediction(model,inp)
        #print("Lasso Predicted Next Day Value : ", lasso_pred)
        return model,lasso_pred

    def ridge_driver(self, x_train,x_test,inp):    
        cls = stock_classifiers()
        #Train the model
        model = cls.ridge(x_train,y_train)
        #Evaluate the model performance
        cls.model_score(model,x_test,y_test)
        # Predict Next Day Value
        ridge_pred = cls.model_prediction(model,inp)
        #print("Ridge Predicted Next Day Value : ", ridge_pred)
        return model,ridge_pred

    def cls_driver(self,x_train,y_train,inp):
        
        print("Linear Regression\n ")
        lr,lr_pred = self.linearregression_driver(x_train,y_train,inp)
        
        print("\nRandom Forest\n ")
        rf,rf_pred = self.randomforest_driver(x_train,x_test,inp)
        
        print("\nDecision Tree\n ")
        dt,dt_pred = self.decisiontree_driver(x_train,x_test,inp)

        print("\nLasso\n ")
        lasso_model,lasso_pred = self.lasso_driver(x_train,x_test,inp)

        print("\nRidge\n ")
        ridge_model,ridge_pred = self.ridge_driver(x_train,x_test,inp)

        val=[lr_pred,rf_pred,dt_pred,lasso_pred,ridge_pred]
        
        cols = ["Linear Regression","Random Forest","Decision Tree","Lasso","Ridge"]
        result = pd.DataFrame(val)
        result.index = cols
        result.columns = ["Predicted Value"]
        
        return result,lr,rf,dt,lasso_model,ridge_model


# In[20]:


path = "D:\\Stock Data\\TRG.csv"
features = ['LDCP','Close']
days_ahead = -1
prp = prepare_stockdata()
df,x_train, x_test, y_train, y_test,X_df = prp.preparation_driver(path,features,days_ahead)
df.tail(10)


# In[21]:


inp = [[118.40,118.41,115.02,103.23,-4.78]]
print(inp)
clf = classification_drivers()
result,lr,rf,dt,lasso_model,ridge_model = clf.cls_driver(x_train,y_train,inp)
#lasso,lasso_pred=lasso_driver(x_train,x_test,inp)
#print(lasso_pred)
print(result)


# In[ ]:





# In[ ]:




