大家好，我是 Wkalpha  
目前正在邁向資料科學家的旅途  
以下是我關於 Kaggle 的建模經驗分享  
本篇會以 House Prices 為例  
有任何問題歡迎來信指教 wkalphakao@gmail.com

# 建模步驟
1. <a href="https://github.com/Wkalpha/pythonKaggleHousePrice/blob/master/README.md#%E8%BC%89%E5%85%A5%E8%B3%87%E6%96%99">載入資料<a>  
   沒有資料，沒有模型  
2. <a href="https://github.com/Wkalpha/pythonKaggleHousePrice/blob/master/README.md#%E6%8E%A2%E7%B4%A2%E5%BC%8F%E8%B3%87%E6%96%99%E5%88%86%E6%9E%90eda">探索式資料分析(EDA)<a>  
   將資料視覺化，幫助我們可以了解資料的樣貌，藉以協助我們建立更好的模型
3. <a href="https://github.com/Wkalpha/pythonKaggleHousePrice/blob/master/README.md#%E8%B3%87%E6%96%99%E9%A0%90%E8%99%95%E7%90%86">資料預處理<a>  
   遺漏值填補：平均數、中位數、眾數；若只有一小部分缺失，可以考慮刪除  
   資料型態轉換：大部分模型無法處理類別的資料，需要將其轉換為數值，常見的方法有 One-hot Encoding  
4. <a href="https://github.com/Wkalpha/pythonKaggleHousePrice/blob/master/README.md#%E7%89%B9%E5%BE%B5%E5%B7%A5%E7%A8%8B">特徵工程<a>  
   非常重要的一環，常聽到「特徵決定模型的上限，演算法只是不斷在逼近這個上限」 
5. <a href="https://github.com/Wkalpha/pythonKaggleHousePrice/blob/master/README.md#%E8%A8%93%E7%B7%B4%E6%A8%A1%E5%9E%8B">訓練模型<a>  
   特徵準備妥當之後，便可開始進行建模
6. <a href="https://github.com/Wkalpha/pythonKaggleHousePrice/blob/master/README.md#%E8%AA%BF%E6%95%B4%E6%A8%A1%E5%9E%8B%E5%8F%83%E6%95%B8">調整模型參數<a>  
   模型的參數很多，之間的組合也有許多變化，除了手動調整以外，還可以透過 Grid Search 自動尋找合適的參數  

# 載入資料  
    import pandas as pd
    train = pd.read_csv('Desktop/train.csv')
    test  = pd.read_csv('Desktop/test.csv')

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
# 特徵工程  
# 訓練模型  
# 調整模型參數  
