# -*- coding: utf-8 -*-
"""Bai 1: Cai dat Linear Regression tu scratch bang Normal Equation.

So sanh he so va R^2 voi sklearn.linear_model.LinearRegression
tren du lieu California Housing.
"""

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


class MyLinearRegression:
    """Linear Regression tu cai dat bang cong thuc Normal Equation."""

    def fit(self, X, y):
        # them cot 1 vao X de hoc duoc he so bias (intercept)
        X_bias = np.hstack([np.ones((X.shape[0], 1)), X])
        # cong thuc dong: theta = (X^T X)^-1 X^T y
        theta = np.linalg.inv(X_bias.T @ X_bias) @ X_bias.T @ y
        self.intercept_ = theta[0]
        self.coef_ = theta[1:]
        return self

    def predict(self, X):
        return X @ self.coef_ + self.intercept_


data = fetch_california_housing()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


my_model = MyLinearRegression()
my_model.fit(X_train, y_train)
my_pred = my_model.predict(X_test)
my_r2 = r2_score(y_test, my_pred)


sk_model = LinearRegression()
sk_model.fit(X_train, y_train)
sk_pred = sk_model.predict(X_test)
sk_r2 = r2_score(y_test, sk_pred)

print("=== So sanh MyLinearRegression vs sklearn.LinearRegression ===")
print()
print("He so (coefficient):")
print("My model :", np.round(my_model.coef_, 4))
print("Sklearn  :", np.round(sk_model.coef_, 4))
print()
print("Intercept:")
print("My model :", round(my_model.intercept_, 4))
print("Sklearn  :", round(sk_model.intercept_, 4))
print()
print("R^2 tren tap test:")
print("My model :", round(my_r2, 4))
print("Sklearn  :", round(sk_r2, 4))
