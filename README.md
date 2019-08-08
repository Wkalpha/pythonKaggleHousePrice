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

# Let's begin
For our practice, I want to use the classical liner problem <a href="https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview" title="Title">House price</a>   

This is my enviroment:  
OS: Windows 10  
Editor: jupyter  
Language: python  

So we don't need to collect data because kaggle already do this for us, we just need download and use.


