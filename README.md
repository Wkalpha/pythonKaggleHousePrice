大家好，我是 Wkalpha  
目前正在邁向資料科學家的旅途  
以下是我關於 Kaggle 的建模經驗分享  
本篇會以 House Prices 為例  
有任何問題歡迎來信指教 wkalphakao@gmail.com

# 建模步驟
1. <a href="https://github.com/Wkalpha/pythonKaggleHousePrice/blob/master/README.md#%E8%BC%89%E5%85%A5%E8%B3%87%E6%96%99">載入資料<a>  
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

# 載入資料  
    import pandas as pd
    train = pd.read_csv('Desktop/train.csv')
    test  = pd.read_csv('Desktop/test.csv')

# 探索式資料分析(EDA)  
# 資料預處理  
# 特徵工程  
# 訓練模型  
# 調整模型參數  
