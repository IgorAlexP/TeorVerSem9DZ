""""""

import numpy as np

# Заданные значения заработной платы и кредитного скоринга
zp = np.array([35, 45, 190, 200, 40, 70, 54, 150, 120, 110])
ks = np.array([401, 574, 874, 919, 459, 739, 653, 902, 746, 832])

# 1. Коэффициенты линейной регрессии без intercept
X = zp
y = ks
slope = np.sum((X - np.mean(X)) * (y - np.mean(y))) / np.sum((X - np.mean(X)) ** 2)
intercept = np.mean(y) - slope * np.mean(X)

print("Коэффициенты линейной регрессии без intercept:")
print("slope =", round(slope, 2))
print("intercept =", round(intercept, 2))
print("---------------------------------------")
# 2. Коэффициент линейной регрессии с использованием градиентного спуска (без intercept)
learning_rate = 0.0001
num_iterations = 1000

X = zp
y = ks

# Нормализация данных
X_normalized = (X - np.mean(X)) / np.std(X)

# Инициализация начальных значений коэффициента
slope = 0

# Градиентный спуск
for _ in range(num_iterations):
    y_pred = slope * X_normalized
    slope -= learning_rate * np.sum((y_pred - y) * X_normalized) / len(X)

intercept = np.mean(y) - slope * np.mean(X)

print("Коэффициент линейной регрессии с использованием градиентного спуска (без intercept):")
print("slope =", round(slope, 2))
print("intercept =", round(intercept, 2))

# 3. Коэффициенты линейной регрессии с использованием градиентного спуска (с intercept)
learning_rate = 0.0001
num_iterations = 1000

X = zp
y = ks

# Нормализация данных
X_normalized = (X - np.mean(X)) / np.std(X)

# Инициализация начальных значений коэффициентов
slope = 0
intercept = 0

# Градиентный спуск
for _ in range(num_iterations):
    y_pred = slope * X_normalized + intercept
    slope_gradient = np.sum((y_pred - y) * X_normalized) / len(X)
    intercept_gradient = np.sum(y_pred - y) / len(X)
    slope -= learning_rate * slope_gradient
    intercept -= learning_rate * intercept_gradient
print("---------------------------------------")
print("Коэффициенты линейной регрессии с использованием градиентного спуска (с intercept):")
print("slope =", round(slope, 2))
print("intercept =", round(intercept, 2))

"""вывод
Коэффициенты линейной регрессии без intercept:
slope = 2.62
intercept = 444.18
---------------------------------------
Коэффициент линейной регрессии с использованием градиентного спуска (без intercept):
slope = 14.74
intercept = -785.02
---------------------------------------
Коэффициенты линейной регрессии с использованием градиентного спуска (с intercept):
slope = 14.74
intercept = 67.56"""

