# BLENDED_LEARNING
# Implementation of Ridge, Lasso, and ElasticNet Regularization for Predicting Car Price

## AIM:
To implement Ridge, Lasso, and ElasticNet regularization models using polynomial features and pipelines to predict car price.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Start the program.

2.Import required libraries such as pandas, numpy, matplotlib, seaborn, and scikit-learn modules.

3.Load the dataset CarPrice_Assignment.csv using pandas.

4.Convert categorical variables to numerical form using one-hot encoding (pd.get_dummies()).

5.Separate input and output variables:

6.Independent variables (X) → all columns except price.

7.Dependent variable (y) → price.

8.Normalize the data using StandardScaler to scale both X and y values.

9.Split the dataset into training and testing sets using train_test_split().

10.Define regression models (Ridge, Lasso, and ElasticNet) and create a pipeline with PolynomialFeatures and the regression model.

11.Train the models and evaluate performance by calculating Mean Squared Error (MSE), Mean Absolute Error (MAE), and R² Score.

12.Display the results and visualize performance using bar plots for MSE and R² Score, then end the program.

## Program:
```
/*
Program to implement Ridge, Lasso, and ElasticNet regularization using pipelines.
Developed by: Carlton Maximus A
RegisterNumber: 21225040052
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
data = pd.read_csv("CarPrice_Assignment.csv")
data.head()
data = pd.get_dummies(data, drop_first=True)
X = data.drop('price', axis=1)
y = data['price']
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(y.values.reshape(-1,1))
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
models = {
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=1.0),
    "ElasticNet": ElasticNet(alpha=1.0, l1_ratio=0.5)
}
results= {}
for name, model in models.items():
    pipeline= Pipeline([
        ('poly', PolynomialFeatures(degree=2)),
        ('regressor', model)
    ])
pipeline.fit(X_train, y_train)
Predictions = pipeline.predict(X_test)
mse = mean_squared_error(y_test, Predictions)
mae = mean_absolute_error(y_test, Predictions)
r2 = r2_score(y_test, Predictions)
results[name] = {'Mse': mse, 'MAE': mae, 'R2SCORE': r2}
print("Name: Carlton Maximus A")
print("Reg no: 212250400522")
for model_name, metrics in results.items():
    print(f"{model_name} - Mean Squared Error: {metrics['Mse']:.2f}, Mean Absolute Error: {metrics['MAE']:.2f}, R2Score: {metrics['R2SCORE']:.2f}")
results_df = pd.DataFrame(results).T
results_df.reset_index(inplace=True)
results_df.rename(columns={'index': 'Model'}, inplace=True)
plt.figure(figsize=(12,5))
plt.subplot(1, 2, 1)
sns.barplot(x= 'Model',y='Mse', data=results_df,palette='viridis')
plt.title('Mean Squared Error (Mse)')
plt.ylabel('Mse')
plt.xticks(rotation=45)
plt.subplot(1, 2, 2)
sns.barplot(x= 'Model', y= 'R2SCORE', data=results_df, palette='viridis')
plt.title('R2SCORE')
plt.ylabel('R2SCORE')
plt.xticks(rotation=45)
plt.tight_layout
plt.show()
```

## Output:
![simple linear regression model for predicting the marks scored](sam.png)
<img width="1010" height="402" alt="image" src="https://github.com/user-attachments/assets/c3f85029-e92b-4565-af4f-f77790b9fd63" />

<img width="606" height="467" alt="image" src="https://github.com/user-attachments/assets/4c49511c-cb65-401c-8bb0-27a51c545a06" />

## Result:
Thus, Ridge, Lasso, and ElasticNet regularization models were implemented successfully to predict the car price and the model's performance was evaluated using R² score and Mean Squared Error.
