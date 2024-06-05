# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

# dataset read
data = pd.read_csv('Change the directory to read the dataset')

# excluiding the columns 'Year', 'Country', 'Status' and 'Life Life expectancy'
variables = data.columns.difference(['Year', 'Country', 'Status', 'Life expectancy'])
features = data[variables]
target = data['Life expectancy']

#Build the Adaline model
class Adaline:
    def __init__(self, learning_rate=0.0001, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        self.errors_ = []

        for _ in range(self.epochs):
            output = self.activation(self.net_input(X))
            errors = (y - output)
            self.weights += self.learning_rate * X.T.dot(errors)
            self.bias += self.learning_rate * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.errors_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.weights) + self.bias

    def activation(self, X):
        return X

    def predict(self, X):
        return self.activation(self.net_input(X))

# NaN values treatments
features = features.fillna(features.mean())
target = target.fillna(target.mean())

# treinament and test division
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# dates standardization
X_train = (X_train - X_train.min()) / (X_train.max() - X_train.min())
X_test = (X_test - X_test.min()) / (X_test.max() - X_test.min())

# if exist NaN values, exclud them
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)
y_train = y_train.fillna(0)
y_test = y_test.fillna(0)

# treatment and predition with adaline
adaline = Adaline(learning_rate=0.0001, epochs=100)
adaline.fit(X_train.values, y_train.values)

y_train_pred = adaline.predict(X_train.values)
y_test_pred = adaline.predict(X_test.values)

# metrics calculation
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
pearson_train, _ = pearsonr(y_train, y_train_pred)
pearson_test, _ = pearsonr(y_test, y_test_pred)

print(f'Erro médio quadrático (treinamento): {mse_train}')
print(f'Erro médio quadrático (teste): {mse_test}')
print(f'Coeficiente de Pearson (treinamento): {pearson_train}')
print(f'Coeficiente de Pearson (teste): {pearson_test}')
print(f'Pesos da rede Adaline: {adaline.weights}')
print(f'Sesgo da rede Adaline: {adaline.bias}')

# Perfomance Graphic
plt.figure(figsize=(14, 7))

# Treatment Graphic
plt.subplot(1, 2, 1)
plt.scatter(y_train, y_train_pred, color='blue', edgecolor='k', alpha=0.7, s=40, marker='o')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=2)
plt.xlabel('Valor Real')
plt.ylabel('Valor Previsto')
plt.title('Conjunto de Treinamento')

# test Graphic
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_test_pred, color='green', edgecolor='k', alpha=0.7, s=40, marker='o')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Valor Real')
plt.ylabel('Valor Previsto')
plt.title('Conjunto de Teste')

plt.tight_layout()
plt.show()
