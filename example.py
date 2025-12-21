import matplotlib.pyplot as plt
import pandas as pd 


df = pd.read_csv("well_data_test_min.csv")
print(df.columns)

x_data = df["X"]
y_data = df["Y"]
print(x_data, y_data)

ct = 4e-5
k = 5
phi = 0.2
mu = 1  # вязкость, cP
B = 1   # объемный коэффициент
h = df["h"]
L = df["L"]
dP = df["dP"]
Q = df["Q"]
t = df["t"]

# Переводим формулы в безразмерную форму
x_ideal = (0.00864 * k * h * dP) / (mu * Q)
y_ideal = (Q * mu * t) / (24 * phi * ct * h * L**2 * dP)

plt.plot(df["X"], df["Y"], 'b-', label='Фактические данные')
plt.plot(x_ideal, y_ideal, 'g--', label='Идеальная теоретическая кривая')
plt.legend()
plt.grid(True)
plt.show()
