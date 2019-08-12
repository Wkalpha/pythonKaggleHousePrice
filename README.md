# What is common step when build the model?
1. Import data (Of course)  
No data no model
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

    # Select two columns into x
    x = feature[['LotArea','TotalBsmtSF','GrLivArea']]
    # Transform to normal distribution
    y = np.log1p(feature['SalePrice'])
    
    # Split the train data, 80% to train and 20% to validation
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 100)
    
    # max_depth more bigger model more easly overfitting
    xgboost = XGBRegressor(max_depth=6)
    xgboost.fit(x_train,y_train)
    
    submission_id = test['Id']
    submission_x = test[['LotArea','TotalBsmtSF','GrLivArea']]
    submission_y = xgboost.predict(submission_x)
    
    # Because we use log1p to transform saleprice, so we need reverse it by using np.exp function
    expo = np.exp(submission_y)
    
    # First column is Id, second is saleprice
    output = pd.DataFrame({'Id': submission_id,
                           'SalePrice': expo})
                           
    # Out put and submit to kaggle
    output.to_csv('Desktop/submission0809.csv', index=False,float_format ='%f')
    
    # Finally I got 0.23542, this is not enough, because I did't preprocess the data, and only use 3 columns
    # But I just want to quickly build a model
    # Next I will preprocess the data and use some important column to bulid my model
    # Then I want to know the different between old and new model

# 開始吧！Exploratory Data Analysis(EDA)
EDA 主要是將資料以視覺化的方式呈現，譬如長條圖、散佈圖、盒形圖等，幫助我從這些圖表中，思考如何對特徵做處理、結合、創造，進一步來說就是認識資料  
舉例來說，我想了解 SalsePrice 與 GrLivArea 的分布情形  
Python 有很棒的套件可以輕鬆達成

    import matplotlib.pyplot as plt
    plt.scatter(GrLivArea,SalsePrice)

結果：  
  ![image](https://github.com/Wkalpha/pythonKaggleHousePrice/blob/master/pltscatter.png)  
  
可以發現這個圖形並不太符合線性分布，因為有存在異常值 
我擔心異常值會影響到模型精度(非絕對)，因此決定刪除 X 軸大於 4500 的點  
參見下列程式碼  

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
    
    # Delete outlier
    feature = feature[feature.GrLivArea < 4500]
    feature.reset_index(drop=True, inplace=True)
    
    # Select two columns into x
    x = feature[['LotArea','TotalBsmtSF','GrLivArea']]
    # Transform to normal distribution
    y = np.log1p(feature['SalePrice'])
    
    # Split the train data, 80% to train and 20% to validation
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 100)
    
    # max_depth more bigger model more easly overfitting
    xgboost = XGBRegressor(max_depth=6)
    xgboost.fit(x_train,y_train)
    
    submission_id = test['Id']
    submission_x = test[['LotArea','TotalBsmtSF','GrLivArea']]
    submission_y = xgboost.predict(submission_x)
    
    # Because we use log1p to transform saleprice, so we need reverse it by using np.exp function
    expo = np.exp(submission_y)
    
    # First column is Id, second is saleprice
    output = pd.DataFrame({'Id': submission_id,
                           'SalePrice': expo})
                           
    # Out put and submit to kaggle
    output.to_csv('Desktop/submission0809.csv', index=False,float_format ='%f')
    
    # 最後我得到 0.2348，
