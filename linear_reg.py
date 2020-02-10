# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('/Users/aniketkumar/Downloads/empsal.csv')
X = dataset.iloc[:, 6:7].values
y = dataset.iloc[:, 7:8].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()

y_train = sc_y.fit_transform(y_train)
y_test = sc_y.transform(y_test)

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




sample_input = np.array([0]).reshape(-1, 1)

sample_output = regressor.predict(sc_X.transform(sample_input))

print(sc_y.inverse_transform(sample_output))


import pickle 
  
# Save the trained model as a pickle string. 
filename = 'regressor.model'
pickle.dump(regressor, open(filename, 'wb')) 


# Save the X scaler as a pickle string. 
filename = 'scaler_x.model'
pickle.dump(sc_X, open(filename, 'wb')) 

# Save the Y scaler as a pickle string. 
filename = 'scaler_y.model'
pickle.dump(sc_y, open(filename, 'wb')) 
