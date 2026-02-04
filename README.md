# BLENDED_LEARNING
# Implementation-of-Linear-Regression-for-Predicting-Car-Prices
## AIM:
To write a program to predict car prices using a linear regression model and test the assumptions for linear regression.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
 Program to implement linear regression model for predicting car prices and test assumptions.
Developed by: Rohith
RegisterNumber: 25008317
*/
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error , r2_score , mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
df = pd.read_csv('CarPrice_Assignment.csv')
df.head()
X = df[['enginesize','horsepower','citympg','highwaympg']]
y = df['price']
X_train , X_test, y_train , y_test = train_test_split(X,y,test_size =0.2 , random_state =42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled =scaler.transform(X_test)
model = LinearRegression()
model.fit(X_train_scaled,y_train)
y_pred = model.predict(X_test_scaled)
print("Name: Rohith S")
print("Reg. No: 25008317")
print("MODEL COEFFICIENTS:")
for feature,coef in zip(X.columns,model.coef_):
    print(f"{feature:}: {coef:}")
print(f"{'Intercept':}: {model.intercept_:}")
print("\nMODEL PERFORMANCE:")
print(f"{'MSE':}:{mean_squared_error(y_test,y_pred):}")
print(f"{'MAE':}:{mean_absolute_error(y_test,y_pred):}")
print(f"{'RMSE':}:{np.sqrt(mean_squared_error(y_test,y_pred)):}")
print(f"{'R-square':}:{r2_score(y_test,y_pred):}")
plt.figure(figsize=(10,5))
plt.scatter(y_test,y_pred,alpha = 0.6)
plt.plot([y.min(),y.max()],[y.min(),y.max()],'r--')
plt.title("Linearity Check: Actual vs Predicted Prices")
plt.xlabel("Actual Price ($)")
plt.ylabel("Predicted Price ($)")
plt.grid(True)
plt.show()
residuals = y_test - y_pred
dw_test = sm.stats.durbin_watson(residuals)
print(f"\nDurbin-Watson Statistic: {dw_test:.2f}","\n(Values close to 2 indicate no autocorrelation)")
plt.figure(figsize=(10,5))
sns.residplot(x=y_pred, y= residuals , lowess = True , line_kws={'color': 'red'})
plt.title("Homoscedasticity Check : Residuals-vs-Predicted")
plt.xlabel("Predicted Price {$}")
plt.ylabel("Residuals {$}")
plt.grid(True)
plt.show()
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,5))
sns.histplot(residuals , kde=True , ax=ax1)
ax1.set_title("Residuals Distribution")
sm.qqplot(residuals, line='45' , fit = True , ax=ax2)
ax2.set_title("Q-Q Plot")
plt.tight_layout()
plt.show()
```

## Output:
<img width="1024" height="324" alt="image" src="https://github.com/user-attachments/assets/617e9954-c103-4101-9a6e-4c6a660291af" />
<img width="1276" height="590" alt="image" src="https://github.com/user-attachments/assets/ee2209b5-f838-4e39-ae09-9936b9b68917" />
<img width="651" height="102" alt="image" src="https://github.com/user-attachments/assets/c7826670-0e9c-43d0-a413-ca9d7d08a640" />
<img width="1286" height="596" alt="image" src="https://github.com/user-attachments/assets/5342391b-c7a3-4071-b1b1-32a55874d70e" />
<img width="1291" height="540" alt="image" src="https://github.com/user-attachments/assets/0c0cacf9-ab15-4567-9e96-d72764bc31b2" />








## Result:
Thus, the program to implement a linear regression model for predicting car prices is written and verified using Python programming, along with the testing of key assumptions for linear regression.
