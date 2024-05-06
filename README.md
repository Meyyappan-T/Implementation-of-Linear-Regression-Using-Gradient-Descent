# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
#### Step 1. Start
#### Step 2. Get population and profit data for cities and Begin with random guesses for how population influences profit.
#### Step 3. Gradually adjust guesses to minimize the difference between predicted and actual profits.
#### Step 4. Keep adjusting until predictions are close to actual profits.
#### Step 5. Once adjusted, predict profit for new city populations.
#### Step 6. Evaluate how well predictions match actual profits.
#### Step 7. If predictions are off, refine guesses and repeat the process.
#### Step 8. Once satisfied, use the model to predict profits based on population for decision-making.
#### Step 9. Stop

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by:Meyyappan.T 
RegisterNumber:  212223240086
*/
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.1,num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    
    theta=np.zeros(X.shape[1]).reshape(-1,1)
    
    for _ in range(num_iters):
        prediction=(X).dot(theta).reshape(-1,1)
        errors=(prediction-y).reshape(-1,1)
        theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta


data=pd.read_csv("C:/Users/admin/Documents/intro_to_ml_meyyappan/50_Startups.csv")
data.head()

X=(data.iloc[1:,:-2].values)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_scaled=scaler.fit_transform(X1)
Y1_scaled=scaler.fit_transform(y)
print(X)
print(X1_scaled)

theta=linear_regression(X1_scaled,Y1_scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted values: {pre}")
```

## Output:
![image](https://github.com/marcoyoi/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/128804366/2ec3891c-dad8-4521-814c-34672802be3f)


![image](https://github.com/marcoyoi/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/128804366/d530135d-493c-4d70-b8dc-1065de7dd6dd)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
