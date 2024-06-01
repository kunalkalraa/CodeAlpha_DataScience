TASK1: Titanic Classification
--------------------------------------------------------------------------------------------------

# Titanic Classification System

## Overview
This project aims to build a classification system that predicts whether a passenger would have survived the sinking of the Titanic. The system uses various features such as socio-economic status, age, gender,
and more to make predictions.

## Dataset
The dataset used for this project is the Titanic dataset, which contains information about the passengers, including whether they survived or not.

## Features
- **Pclass**: Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)
- **Sex**: Gender (male, female)
- **Age**: Age of the passenger
- **SibSp**: Number of siblings/spouses aboard the Titanic
- **Parch**: Number of parents/children aboard the Titanic
- **Fare**: Passenger fare
- **Embarked**: Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)

## Steps to Build the System

1. **Data Collection and Preparation**:
    - Load the Titanic dataset.
    - Handle missing values.
    - Encode categorical variables.
    - Normalize numerical features.

2. **Feature Selection**:
    - Identify the most relevant features for predicting survival.

3. **Model Selection and Training**:
    - Split the data into training and testing sets.
    - Train a machine learning model (e.g., Logistic Regression, Decision Tree, Random Forest, Gradient Boosting).

4. **Feature Importance Analysis**:
    - Determine the importance of each feature in predicting survival.

5. **Prediction System**:
    - Create a system that takes input features for a passenger and predicts whether they would survive.

## Implementation

The implementation is provided in Python using Jupyter Notebook. The code includes data preprocessing, model training, evaluation, and a prediction function.

### Usage

1. **Install Dependencies**:
    - Ensure you have Python and the necessary libraries installed:
      ```bash
      pip install numpy pandas matplotlib seaborn scikit-learn
      ```

2. **Run the Notebook**:
    - Load the `titanic_classification.ipynb` notebook and execute the cells.

3. **Prediction Function**:
    - Use the `predict_survival` function to predict whether a passenger would survive:
      ```python
      predict_survival(3, 'male', 22, 1, 0, 7.25, 'S')
      ```

## Results

- **Accuracy**: The model's accuracy on the test set.
- **Feature Importance**: Analysis of the most important features for predicting survival.

## Conclusion

This project demonstrates how to build a machine learning classification system using the Titanic dataset. It covers data preprocessing, model training, evaluation, and deployment of a prediction system.


































TASK2: Stock Prediction
-----------------------------------------------------------------------------------------------------------

# Stock Price Prediction Using LSTM

## Overview
This project predicts the stock prices of Apple Inc. (AAPL) using a Long Short-Term Memory (LSTM) network. The LSTM model is a type of recurrent neural network (RNN) suitable for time series forecasting.

## Dataset
The stock price data is fetched from Yahoo Finance using the `yfinance` library. The dataset contains the daily closing prices of AAPL from January 2015 to January 2023.

## Steps to Build the System

1. **Data Collection**:
    - Use the `yfinance` library to download historical stock price data.

2. **Data Preprocessing**:
    - Handle missing values by dropping them.
    - Normalize the stock prices using MinMaxScaler.
    - Create training and testing datasets.
    - Prepare the data for LSTM by creating sequences of stock prices.

3. **Model Building**:
    - Build the LSTM model using Keras.
    - Train the model on the training data.
    - Validate the model on the testing data.

4. **Prediction and Visualization**:
    - Make predictions on the training and testing datasets.
    - Inverse transform the predicted prices to their original scale.
    - Plot the actual and predicted stock prices for visualization.

## Implementation

The implementation is provided in Python using Jupyter Notebook. The code includes data fetching, preprocessing, model building, training, evaluation, and visualization.

### Usage

1. **Install Dependencies**:
    - Ensure you have Python and the necessary libraries installed:
      ```bash
      pip install numpy pandas matplotlib yfinance scikit-learn keras tensorflow
      ```

2. **Run the Notebook**:
    - Load the `stock_price_prediction.ipynb` notebook and execute the cells.

3. **Prediction and Visualization**:
    - The notebook will output the predicted stock prices and plot the actual vs predicted prices.

## Results

- **Train and Test Predictions**: The model's predictions on both the training and testing datasets.
- **Visualization**: Graphical representation of actual vs predicted stock prices.

## Conclusion

This project demonstrates how to build an LSTM-based model for stock price prediction using historical stock price data. It covers data preprocessing, model training, evaluation, and visualization of results.

































TASK3: Predictive Modeling with Linear Regression
-----------------------------------------------------------------------------------------------------------------

# Predictive Modeling with Linear Regression

## Overview
This project demonstrates how to build a predictive model using linear regression to predict a numerical outcome based on one or more features. We use the Boston Housing dataset, which contains information
about various features of houses in Boston along with their prices.

## Dataset
The dataset used for this project is the Boston Housing dataset. It includes various features such as:
- **CRIM**: Per capita crime rate by town
- **ZN**: Proportion of residential land zoned for lots over 25,000 sq. ft.
- **INDUS**: Proportion of non-retail business acres per town
- **CHAS**: Charles River dummy variable (1 if tract bounds river; 0 otherwise)
- **NOX**: Nitric oxide concentration (parts per 10 million)
- **RM**: Average number of rooms per dwelling
- **AGE**: Proportion of owner-occupied units built prior to 1940
- **DIS**: Weighted distances to five Boston employment centers
- **RAD**: Index of accessibility to radial highways
- **TAX**: Full-value property tax rate per $10,000
- **PTRATIO**: Pupil-teacher ratio by town
- **B**: 1000(Bk - 0.63)^2 where Bk is the proportion of black residents by town
- **LSTAT**: Percentage of lower status of the population
- **PRICE**: Median value of owner-occupied homes in $1000s (target variable)

## Steps to Build the System

1. **Data Collection and Preparation**:
    - Load the Boston Housing dataset.
    - Display the first few rows of the dataset.
    - Handle any missing values.

2. **Exploratory Data Analysis (EDA)**:
    - Display summary statistics of the dataset.
    - Visualize the relationships between features and the target variable using pair plots.
    - Compute and visualize the correlation matrix.

3. **Data Preparation**:
    - Define features and target variable.
    - Split the data into training and testing sets.

4. **Model Building and Training**:
    - Create and train the Linear Regression model using the training data.

5. **Model Evaluation**:
    - Make predictions on the test set.
    - Evaluate the model using metrics such as Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R2) score.

6. **Results Visualization**:
    - Create scatter plots to visualize the actual vs predicted values.
    - Create residual plots to visualize the residuals.

## Implementation

The implementation is provided in Python using Jupyter Notebook. The code includes data loading, preprocessing, exploratory data analysis, model training, evaluation, and visualization of results.

### Usage

1. **Install Dependencies**:
    - Ensure you have Python and the necessary libraries installed:
      ```bash
      pip install numpy pandas matplotlib seaborn scikit-learn
      ```

2. **Run the Notebook**:
    - Load the `linear_regression_modeling.ipynb` notebook and execute the cells.

3. **Evaluate the Model**:
    - The notebook will output the evaluation metrics and visualizations.

## Results

- **Mean Squared Error (MSE)**: Indicates the average squared difference between the actual and predicted values.
- **Root Mean Squared Error (RMSE)**: The square root of MSE, providing error in the same units as the target variable.
- **R-squared (R2) Score**: Indicates the proportion of the variance in the dependent variable that is predictable from the independent variables.

## Conclusion

This project demonstrates how to build a linear regression model for predictive modeling using the Boston Housing dataset. It covers data preprocessing, exploratory data analysis, model training, evaluation, 
and visualization of results.



