#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install mplfinance


# In[24]:


import pandas as pd
#import yfinance as yf
import mplfinance as mf
import matplotlib.pyplot as plt


# In[25]:


# Loading data with date column

class plot_data:
    
    def __init__(self):
      self.stock_df = ""

    def load_stock_with_date(self,path):
      self.stock_df = pd.read_csv(path,sep=',')
      #stock.columns=['Date','Open', 'High', 'Low', 'Avg.','Close', 'Volume']
      self.stock_df['Date']=pd.to_datetime(stock_df.Date)
      self.stock_df.index = self.stock_df['Date']
      self.stock_df = self.stock_df.sort_index(ascending=False)
      self.stock_df['Volume'] = self.stock_df['Turnover']
      # Remove column name Date and Volume
      self.stock_df = self.stock_df.drop(['Date'], axis = 1)

      self.stock_df = self.stock_df.drop(['Turnover'], axis = 1)
      self.stock_df = self.stock_df.drop(['Symbol'], axis = 1)
      self.stock_df = self.stock_df.drop(['Name'], axis = 1)
      self.stock_df = self.stock_df.drop(['Avg.'], axis = 1)
      self.stock_df = self.stock_df.drop(['LDCP'], axis = 1)
      return self.stock_df
    
    def plot_candlestick(self,levels):
      # Plot N rows
      stockdata = self.stock_df[:260].sort_values(by='Date',ascending=True)
      mf.plot(stockdata, type = 'candle', style =  'charles',title = 'Stock Analysis with CandleStick Chart',
            ylabel = 'Prices', figratio=(12,8),mav=(3,6,9),volume=True, hlines=dict(hlines=levels,colors=['g','r'],linestyle='-.'))

    def plot_stock(self,features):
      self.stock_df[features].plot()
      plt.legend()


# In[26]:


#path = "D:\\Stock Data\\TRG.csv"

#stock_plt = plot_data()
#stock = stock_plt.load_stock_with_date(path)





# In[27]:


#levels = [80,120,140,160,180]
#stock_plt.plot_candlestick(stock,levels)


# In[28]:


#features = ['Close','Open']
#stock_plt.plot_stock(stock,features)


# In[ ]:




