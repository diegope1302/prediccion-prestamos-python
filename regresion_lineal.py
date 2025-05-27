import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Cargar dataset
df = pd.read_csv('dataBasePrestDigital.csv', sep=';')

# Filtrar filas con monto > 0
df_filtrado = df[df['promSaldoPrest3Um'] > 0]

# Calcular frecuencia total por cliente
frecuencia = df.groupby('cliente').size().reset_index(name='frecuencia_transacciones')

# Calcular monto promedio solo para montos > 0
monto_promedio = df_filtrado.groupby('cliente')['promSaldoPrest3Um'].mean().reset_index(name='monto_promedio')

# Unir tablas
df_modelo = pd.merge(frecuencia, monto_promedio, on='cliente', how='left')

# Rellenar NaN con 0
df_modelo['monto_promedio'] = df_modelo['monto_promedio'].fillna(0)

# Filtrar para regresión solo clientes con monto > 0 (opcional, para mejor ajuste)
df_modelo_reg = df_modelo[df_modelo['monto_promedio'] > 0]

# Definir variables X y y
X = df_modelo_reg[['frecuencia_transacciones']].values  # Predictor (2D array)
y = df_modelo_reg['monto_promedio'].values  # Variable a predecir

# Crear y entrenar el modelo de regresión lineal simple
modelo = LinearRegression()
modelo.fit(X, y)

# Coeficiente (pendiente) y ordenada al origen
print(f'Coeficiente: {modelo.coef_[0]}')
print(f'Intercepto: {modelo.intercept_}')

# Predecir valores para la línea
y_pred = modelo.predict(X)

# Graficar
plt.scatter(X, y, color='blue', alpha=0.5, label='Datos reales')
plt.plot(X, y_pred, color='red', label='Regresión lineal')
plt.xlabel('Frecuencia de transacciones')
plt.ylabel('Monto promedio préstamo')
plt.title('Regresión Lineal Simple')
plt.legend()
plt.show()
