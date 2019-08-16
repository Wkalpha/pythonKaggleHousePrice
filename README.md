大家好，我是 Wkalpha  
目前正在邁向資料科學家的旅途  
以下是我關於 Kaggle 的建模經驗分享  
本篇會以 <a href="https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview">House Prices<a> 為例  
有任何問題歡迎來信指教 wkalphakao@gmail.com  

<a href="https://github.com/Wkalpha/pythonKaggleHousePrice/blob/master/README.md#%E5%AE%8C%E6%95%B4%E7%A8%8B%E5%BC%8F%E7%A2%BC">完整程式碼<a>  

# 建模步驟
1. <a href="https://github.com/Wkalpha/pythonKaggleHousePrice/blob/master/README.md#%E8%BC%89%E5%85%A5%E8%B3%87%E6%96%99">載入資料<a>  
   沒有資料，沒有模型  
2. <a href="https://github.com/Wkalpha/pythonKaggleHousePrice/blob/master/README.md#%E6%8E%A2%E7%B4%A2%E5%BC%8F%E8%B3%87%E6%96%99%E5%88%86%E6%9E%90eda">探索式資料分析(EDA)<a>  
   將資料視覺化，幫助我們可以了解資料的樣貌，藉以協助我們建立更好的模型
3. <a href="https://github.com/Wkalpha/pythonKaggleHousePrice/blob/master/README.md#%E8%B3%87%E6%96%99%E9%A0%90%E8%99%95%E7%90%86">資料預處理<a>  
   遺漏值填補：平均數、中位數、眾數；若只有一小部分缺失，可以考慮刪除  
   資料型態轉換：大部分模型無法處理類別的資料，需要將其轉換為數值，常見的方法有 One-hot Encoding  
4. <a href="https://github.com/Wkalpha/pythonKaggleHousePrice/blob/master/README.md#%E7%89%B9%E5%BE%B5%E5%B7%A5%E7%A8%8B">特徵工程<a>  
   非常重要的一環，常言道「特徵決定模型的上限，演算法只是不斷在逼近這個上限」  
   有幾種方法  
   1.混合其他特徵成為新的特徵  
   2.對特徵做算數  
   3.切割特徵  
   
5. <a href="https://github.com/Wkalpha/pythonKaggleHousePrice/blob/master/README.md#%E8%A8%93%E7%B7%B4%E6%A8%A1%E5%9E%8B">訓練模型<a>  
   資料及特徵準備妥當之後，便可開始進行建模  
   這邊我會用多種模型進行預測，取各種模型的最佳解混合  
   
6. <a href="https://github.com/Wkalpha/pythonKaggleHousePrice/blob/master/README.md#%E8%AA%BF%E6%95%B4%E6%A8%A1%E5%9E%8B%E5%8F%83%E6%95%B8">調整模型參數<a>  
   模型的參數很多，之間的組合也有許多變化，除了手動調整以外，還可以透過 Grid Search 自動尋找合適的參數  

# 載入資料  
    import pandas as pd
    train = pd.read_csv('Desktop/train.csv')
    test  = pd.read_csv('Desktop/test.csv')
    
    # 把訓練資料與測試資料合併做預處理，之後再分開
    all_df = pd.concat((train, test), axis=0)

# 探索式資料分析(EDA)  
  將資料以視覺化方式呈現，散佈圖、直條圖、盒鬚圖等，幫助快速了解資料的型態，以利後續步驟執行  

1. 透過散佈圖觀察 GrLivArea 與 SalePrice  

       import matplotlib.pyplot as plt
       plt.scatter(GrLivArea,SalePrice)
       
2. 透過盒鬚圖觀察 OverallQual 與 SalePrice  
       
       import seaborn as sns
       data = pd.concat([train['OverallQual'],train['SalePrice']],axis = 1)
       sns.boxplot(x="OverallQual", y="SalePrice", data=data)

3. 透過長條圖觀察  SalePrice  

       import seaborn as sns
       sns.distplot(train['SalePrice'])
       
# 資料預處理  
  大部分的模型無法對缺失值、異常值做處理，需要將其填補平均、中位、眾數，甚至必要的話可直接刪除該欄位或該列  
  
  可以透過語法統計各欄位的缺失值數量  
  
    train.isnull().sum().sort_values(ascending=False)
    
  根據之前匯入的資料，統計後如下  
  
      PoolQC           1453
      MiscFeature      1406
      Alley            1369
      Fence            1179
      FireplaceQu       690
      LotFrontage       259
      GarageCond         81
      GarageType         81
      GarageYrBlt        81
      GarageFinish       81
      GarageQual         81
      BsmtExposure       38
      BsmtFinType2       38
      BsmtFinType1       37
      BsmtCond           37
      BsmtQual           37
      MasVnrArea          8
      MasVnrType          8
      Electrical          1
   
   至於欄位要怎麼填補缺失值，需要根據 Domain know how 來判斷，這邊就不一一贅述  
   
# 特徵工程  
# 訓練模型  
# 調整模型參數  
# 完整程式碼  
    # Start here
