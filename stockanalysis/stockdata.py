#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# In[11]:


class prepare_stockdata: 
    def getstockdata(self,path):
        df = pd.read_csv(path)
        df['Date'] = pd.to_datetime(df['Date'])
        df.sort_values(by=['Date'], inplace=True)
        df.index = df['Date']
        df = df.drop(['Date'], axis = 1)
        return df

    def getstockfeatures(self,df,features):
        df1 = pd.DataFrame(df[features])
        return df1

        
    def add_features(self,df):
        df['7DayAve'] = df['Close'].rolling(7).mean()
        df['15DayAve'] = df['Close'].rolling(15).mean()
        df["Day_Perc_Change"] = df["Close"].pct_change()*100
        #add support_resistance levels
        df = self.support_resistance(df)
        df.dropna(axis = 0, inplace = True)
        return df
    
    def support_resistance(self,data_ohlc):
        # Pivot Zone Calculation

        data_ohlc['Pivot'] = (data_ohlc["High"] + data_ohlc['Low'] + data_ohlc['Close'])/3
        data_ohlc['R1'] = (2 * data_ohlc['Pivot']) - data_ohlc['Low']
        data_ohlc['S1'] = (2 * data_ohlc['Pivot']) - data_ohlc['High']
        data_ohlc['R2'] = (data_ohlc['Pivot']) + (data_ohlc['High'] - data_ohlc['Low'])
        data_ohlc['S2'] = (data_ohlc['Pivot']) - (data_ohlc['High'] - data_ohlc['Low'])
        data_ohlc['R3'] = (data_ohlc['R1']) + (data_ohlc['High'] - data_ohlc['Low'])
        data_ohlc['S3'] = (data_ohlc['S1']) - (data_ohlc['High'] - data_ohlc['Low'])
        data_ohlc['R4'] = (data_ohlc['R3']) + (data_ohlc['R2'] - data_ohlc['R1'])
        data_ohlc['S4'] = (data_ohlc['S3']) - (data_ohlc['S1'] - data_ohlc['S2'])
        return data_ohlc


    #days_ahead can be set of -1 for next day prediction
    #df is the dataframe containing features
    def set_predictioncol(self,df,days_ahead):
        df['Prediction'] = df[['Close']].shift(days_ahead)
        return df

    def prepare_dataset(self,df,days_ahead):
        #Create a data set X and convert it into numpy array , which will be having actual values
        X = np.array(df.drop(['Prediction'],1))
        #Remove the last N rows

        X = X[:days_ahead]

        #print('X : \n',X)
        # Create a dataset y which will be having Predicted values and convert into numpy array
        y = np.array(df['Prediction'])
        # Remove Last 15 rows
        y = y[:days_ahead]
        #print(y)
        return X, y
    
    def preparation_driver(self,path,features,days_ahead):
        

        #Step 1 : Load Data
        df = self.getstockdata(path)


        #Step 2: Add relevant features
        df = self.add_features(df)
        
        #Step 3: Select relevant Features
        df = self.getstockfeatures(df,features)

        #Step 4: Setup/Create prediction column
        df = self.set_predictioncol(df,days_ahead)

        ##Step 5: Prepare dataset for classification
        X, y = self.prepare_dataset(df,days_ahead)

        #Step 6: Split the data into train and test with 90 & 10 % respectively
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        #To return X part of dataframe without movemnet
        print(df.head())
        X_df = np.array(df.drop([['Prediction','Close']]))

        return df,x_train, x_test, y_train, y_test,X_df



# In[12]:


path = "D:\\Stock Data\\TRG.csv"
features = ['LDCP','Close']
days_ahead = -1
prp = prepare_stockdata()
df,x_train, x_test, y_train, y_test,X_df = prp.preparation_driver(path,features,days_ahead)
df.tail(10)


# In[ ]:




