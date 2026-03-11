import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


df = pd.read_csv('student_data.csv')

# 1. Linear Regression
X = df[['Study_Hours']]
y = df['Marks']
model_lr = LinearRegression().fit(X, y)
print("Linear Prediction for 9 hours:", model_lr.predict([[9]]))

# 2. Polynomial Regression (Interaction Features)
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(df[['Study_Hours', 'Sleep_Hours']])
model_poly = LinearRegression().fit(X_poly, y)
print("Polynomial Coefficients:", model_poly.coef_)
