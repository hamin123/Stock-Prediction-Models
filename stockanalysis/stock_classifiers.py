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

class stock_classifiers:
    def linear_regression(self,x_train,y_train):
        # Linear Regression Model
        lr = LinearRegression()
        # Train the model
        lr.fit(x_train, y_train)
        return lr

    def lr_score(self,model,x_test,y_test):
        # The best possible score is 1.0
        lr_confidence = model.score(x_test, y_test)
        print("Linear Regression Confidence : ",lr_confidence)

    def model_prediction(self,model,inp):
        inp_array = np.array(inp)
        pred = model.predict(inp_array)
        return pred


    def svm(self,x_train,y_train):
        from sklearn.svm import SVR
        # SVM Model
        svr = SVR(kernel='rbf', C=1e3, gamma=0.1)
        # Train the model 
        svr.fit(x_train, y_train)
        return svr

    def model_score(self,model,x_test,y_test):
        # The best possible score is 1.0
        model_confidence = model.score(x_test, y_test)
        print("model confidence: ", model_confidence)
        return model_confidence

    def model_error(self,y_test,y_pred):

        print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
        print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

    def feature_scaling(self,x_train,x_test):
        sc = StandardScaler()
        x_train_sc = sc.fit_transform(x_train)
        x_test_sc = sc.transform(x_test)
        return sc,x_train_sc,x_test_sc
    
    def decision_tree(self,x_train,y_train):
        # Initialize Decision Tree Model Model
        dtr = DecisionTreeRegressor(random_state=0)   
        #fit the data
        model = dtr.fit(x_train, y_train)
        return model

    

    def random_forest(self, x_train,y_train,n_estimators):
        rf = RandomForestRegressor(n_estimators, random_state=0)
        rf.fit(x_train, y_train)
        return rf
    
    def lasso(self, x_train, y_train):
    
        lasso_clf = LassoCV(n_alphas=1000, max_iter=3000, random_state=0)
        model = lasso_clf.fit(x_train,y_train)
        #prediction = model.predict(validation_x)
        return model


    def ridge(self, x_train, y_train):
        """
        This method uses to train the data.
        args:
            x_train : feature training set
            y_train : target training set
            validation_x : validation feature set
        return:
            model : returns the trained model
        """
        ridge_clf = RidgeCV(gcv_mode='auto')
        model = ridge_clf.fit(x_train,y_train)
        #     prediction = ridge_model.predict(validation_x)
        return model