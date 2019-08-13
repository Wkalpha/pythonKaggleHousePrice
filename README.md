# 常見的建模步驟
1. 載入資料  
   沒有資料，沒有模型  
2. 探索式資料分析(EDA)  
   將資料視覺化，幫助我們可以了解資料的樣貌，藉以協助我們建立更好的模型
3. 資料預處理  
   遺漏值填補：平均數、中位數、眾數；若只有一小部分缺失，可以考慮刪除  
   資料型態轉換：大部分模型無法處理類別的資料，需要將其轉換為數值，常見的方法有 One-hot Encoding  
4. 特徵工程  
   非常重要的一環，常聽到「特徵決定模型的上限，演算法只是不斷在逼近這個上限」 
5. 訓練模型  
   特徵準備妥當之後，便可開始進行建模
6. 調整模型參數  
   模型的參數很多，之間的組合也有許多變化，除了手動調整以外，還可以透過 Grid Search 決定合適的參數  

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
完整程式碼如下  

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
    
    # 第一個欄位是 ID，再來是 SalePrice
    output = pd.DataFrame({'Id': submission_id,
                           'SalePrice': expo})
                           
    # 輸出為 submission 檔案，並提交給 Kaggle
    output.to_csv('Desktop/submission0809.csv', index=False,float_format ='%f')
    
最後我得到 0.2348，雖然有進步，但仍然不夠，因此我決定增加一些特徵幫助建立模型  
增加特徵之前我還想知道 OverallQual 和 SalePrice 的關係，因為 OverallQual 是類別變數，所以我打算用盒鬚圖來看  
完整程式碼如下  
    
    # Box plot OverallQual and SalePrice
    data = pd.concat([y, x['OverallQual']], axis=1)
    f, ax = plt.subplots(figsize=(8, 6))
    fig = sns.boxplot(x="OverallQual", y="SalePrice", data=data)
    fig.axis(ymin=10, ymax=14)
    
結果：  
![image](https://github.com/Wkalpha/pythonKaggleHousePrice/blob/master/Boxplot.png)
