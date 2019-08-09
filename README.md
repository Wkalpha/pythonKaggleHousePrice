# What is common step when build the model?
1. Import data (Of course)  
Train and test data should be import.  
2. Exploratory data analysis(EDA) your data  
Understand data that help you how to preprocess data.  
3. Preprocess data(Garbage in garbage out)  
Fill NaN : Model can't handle NaN data, so we need to fill that with median or zero or something.
Delete outlier : Outlier data will reduce model performance, so we need decide delete or not.
Convert data type : Linear model usually can't handle text feature, so we need conver that into int.  
4. Feature engineering  
Most important thing in data science, it decide your model's performance.  
5. Training model  
When prepare all need data, we can use it to training our model like XGBoostRegressor.  
6. Tuneing model paramater  
This step can imporve your model performance.

# Let's quickly bulid XGBoost model!
For our practice, I want to use the classical liner problem <a href="https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview" title="Title">House price</a>   

This is my enviroment:  
OS: Windows 10  
Editor: jupyter  
Language: python  

So we don't need to collect data because kaggle already do this for us, we just need download and use.  

I want to quickly bulid XGBoost model and submit to kaggle...see below

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from xgboost import XGBRegressor
    from sklearn.model_selection import train_test_split
    
    train = pd.read_csv('Desktop/house-prices-advanced-regression-techniques/train.csv')
    test  = pd.read_csv('Desktop/house-prices-advanced-regression-techniques/test.csv')
    pd.set_option('display.max_columns', None) # show all columns
    pd.set_option('display.max_row', None) # show all rows  
    
    # Drop 'Id' column, because we don't need this for our model
    feature = train.drop(['Id',],axis = 1)

    feature = feature[feature.GrLivArea < 4500]
    feature.reset_index(drop=True, inplace=True)

    # Select two columns into x
    x = feature[['LotArea','TotalBsmtSF']]
    # Transform to normal distribution
    y = np.log1p(feature['SalePrice'])
    
    # Split the train data, 80% to train and 20% to validation
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 100)
    
    # max_depth more bigger model more easly overfitting
    xgboost = XGBRegressor(max_depth=6)
    xgboost.fit(x_train,y_train)
    
    # Put test's column into submission
    submission_x = test[['LotArea','TotalBsmtSF']]
    xgboost.predict(submission_x)
    
    submission_id = test['Id']
    submission_x = test[['LotArea','TotalBsmtSF']]
    submission_y = xgboost.predict(submission_x)
    
    # Because we use log1p to transform saleprice, so we need reverse by using np.exp
    expo = np.exp(submission_y)
    
    # First column is Id, second is saleprice
    output = pd.DataFrame({'Id': submission_id,
                           'SalePrice': expo})
                           
    # Out put and submit to kaggle
    output.to_csv('Desktop/submission0809.csv', index=False,float_format ='%f')
    
    # Finally my score is 0.30590, this score is not good, because I did't preprocess the data, use few column...etc,
    # I just want to quickly build a model
    # Next I will preprocess the data and use some important column to bulid my model
    # Then I want to know the different between old and new model

# Let's begin
    
