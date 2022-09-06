Churn-Prediction-With-Telcom-Data
Comparing Logistic Regression, Decision Tree and Random Forest for Churn Prediction
For any company providing services with recurring billling understanding and predicting customer churn is important. And with adequate data, machine learning and predict those customers who are likley to jump ship or seek services elsewhere.
This has a significant impact on the company's bottom line.
This exercise utlised the data from the Telcom Churn Dataset
https://www.kaggle.com/datasets/mnassrib/telecom-churn-datasets
The codes tested and compared Logistic Regression, Decision Tree and Random Forest for this classification problem
Analysis were carried out in stages
1. Data PRocessing
This includes: 
    reading the datasets, 
    combining them, 
    checking for missing values, 
    exploratory data analysis (EDA), 
    generating correlation tables, 
    removal of redundant variables, 
    exploration of the categorial variables,  and
    splitting the dataset into training and testing sets.
2. Prediction exercise
a. Fitting the Logistic Regression Model
    Computing the variance table for examination of feature importance
    Assessing the predictive ability of the Logistic Regression model
    RESULT: "Logistic Regression Accuracy 0.854354354354354
    Generating the Confusion Matrix
    Computing the Odd ratios
    RESULTS: Intl plan, Intl call, voice mail plan and State VT greatly increase the odds of Churn 8 - 9 folds
b. Decision Tree
    Using those variable which increased the odds the most for simplicity sake
        State+International.plan+Voice.mail.plan+ Total.intl.calls
    