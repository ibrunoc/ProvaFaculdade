import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = {
    "Ano": [2023, 2022, 2021, 2020, 2019, 2018, 2016, 2015, 2014, 2013],
    "Mulheres": [34, 28, 25, 16, 35, 27, 26, 8, 11, 15]
}

df = pd.DataFrame(data)

df = df.sort_values(by="Ano").reset_index(drop=True)

X = df["Ano"].values.reshape(-1, 1)
y = df["Mulheres"].values

model = LinearRegression()
model.fit(X, y)

year_2024 = np.array([[2024]])
predicted_2024 = model.predict(year_2024)[0]

y_pred = model.predict(X)

plt.figure(figsize=(10, 6))
plt.plot(df["Ano"], y, label="Número de Mulheres na TI", color="blue", marker='o', linestyle='-', linewidth=2) 
plt.plot(df["Ano"], y_pred, label="Linha de regressão linear", color="red", linestyle='--', linewidth=2) 
plt.scatter(2024, predicted_2024, color="green", marker="x", s=100, label=f"Predição para 2024: {predicted_2024:.2f}") 

plt.xticks(np.arange(2013, 2025, 1), rotation=45)
plt.xlabel("Ano")
plt.ylabel("Número de Mulheres na área de TI")
plt.title("Previsão do número de mulheres na área de TI")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

df["Predição"] = y_pred
df.loc[len(df.index)] = [2024, None, predicted_2024]
df.to_excel("predicao_mulheres_ti_2024_final.xlsx", index=False)

print(f"Previsao para 2024: {predicted_2024:.2f}")
print(f"Erro medio quadratico (MSE): {mse:.2f}")
print(f"Coeficiente de determinacao (R²): {r2:.2f}")