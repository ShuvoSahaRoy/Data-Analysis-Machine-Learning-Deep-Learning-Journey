# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# dataset = pd.read_csv(r'C:\Users\sshuv\Downloads\R&D Farm Cattle Measurement dataset.csv')
# X = dataset.iloc[:,2:8].values
# y = dataset.loc[:, 'Live Weight'].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)


# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# evaluate model
# R^2
score = regressor.score(X_test, y_test)

from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

#display adjusted R-squared
adjusted_r2 = 1 - (((1-r2)*(len(y_test)-1))/(len(y_test)-X.shape[1]-1))

"""
Differences among these evaluation metrics
Mean Squared Error(MSE) and Root Mean Square Error penalizes the large prediction errors vi-a-vis Mean Absolute Error (MAE). 
However, RMSE is widely used than MSE to evaluate the performance of the regression model with other random models as it has
the same units as the dependent variable (Y-axis).
MSE is a differentiable function that makes it easy to perform mathematical operations in comparison to a non-differentiable
 function like MAE. Therefore, in many models, RMSE is used as a default metric for calculating Loss Function despite being
 harder to interpret than MAE.
The lower value of MAE, MSE, and RMSE implies higher accuracy of a regression model. However, a higher value of R square is
 considered desirable.
R Squared & Adjusted R Squared are used for explaining how well the independent variables in the linear regression model 
explains the variability in the dependent variable. R Squared value always increases with the addition of the independent
variables which might lead to the addition of the redundant variables in our model. However, the adjusted R-squared solves
this problem.
Adjusted R squared takes into account the number of predictor variables, and it is used to determine the number of 
independent variables in our model. The value of Adjusted R squared decreases if the increase in the R square by the
 additional variable isn’t significant enough.
For comparing the accuracy among different linear regression models, RMSE is a better choice than R Squared.
Conclusion

Both RMSE and R- Squared quantifies how well a linear regression model fits a dataset. The RMSE tells how well a
 regression model can predict the value of a response variable in absolute terms while R- Squared tells how well 
 the predictor variables can explain the variation in the response variable.
"""




