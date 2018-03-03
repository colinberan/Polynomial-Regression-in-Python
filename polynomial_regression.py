# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values # This format ensures that the X variable is considered a matrix
y = dataset.iloc[:, 2].values

# Fitting Polynomial Regression to the dataset
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)

# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Position Level vs Salary')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results
X_grid = np.arange(min(X), max(X), 0.1)     # Set a smaller step size for smoother curve
X_grid = X_grid.reshape((len(X_grid), 1))   # Set X_grid as a matrix, rows for every step in X_grid, 1 column
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Position Level vs Salary')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Polynomial Regression
lin_reg.predict(poly_reg.fit_transform(6.5))