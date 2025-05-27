import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np

# --- Cargar dataset ---
df = pd.read_csv('dataBasePrestDigital.csv', sep=';')

# Filtrar filas con monto > 0 para tener datos consistentes
df_filtrado = df[df['promSaldoPrest3Um'] > 0]

# Calcular frecuencia de transacciones (total)
frecuencia = df.groupby('cliente').size().reset_index(name='frecuencia_transacciones')

# Promedio saldo préstamo 3 meses para monto objetivo y como feature
prom_saldo = df_filtrado.groupby('cliente')['promSaldoPrest3Um'].mean().reset_index(name='promSaldoPrest3Um')

# Para 'rngEdad', convertir rangos a números (p.ej: '<35', '35-45', '45-55' etc.)
# Primero miramos valores únicos para convertirlos
print("Valores únicos en rngEdad:", df['rngEdad'].unique())

# Asumiendo rangos comunes, definimos mapa manual (ajusta según tu dataset real)
rango_a_num = {
    '<35': 30,
    '<35-45]': 40,
    '<45-55]': 50,
    '<55-65]': 60,
    '>65': 70,
}

# Convertir valores categóricos a numéricos
df['rngEdad_num'] = df['rngEdad'].map(rango_a_num)

# En caso de valores no mapeados, poner NaN y luego quitar
df = df.dropna(subset=['rngEdad_num'])

# Calcular edad promedio por cliente (ya numérico)
edad = df.groupby('cliente')['rngEdad_num'].mean().reset_index(name='rngEdad')

# Unir todas las features en un dataframe
df_features = frecuencia.merge(prom_saldo, on='cliente', how='inner') \
                       .merge(edad, on='cliente', how='inner')

# Variables predictoras para regresión lineal (sin incluir monto como feature)
X_reg = df_features[['frecuencia_transacciones', 'rngEdad']].values
y_reg = df_features['promSaldoPrest3Um'].values

# División para regresión lineal (80/20)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Modelo regresión lineal
model_lr = LinearRegression()
model_lr.fit(X_train_reg, y_train_reg)

# Predecir regresión lineal
y_pred_reg = model_lr.predict(X_test_reg)

# Gráfico regresión lineal: reales vs predichos
plt.figure(figsize=(6,6))
plt.scatter(y_test_reg, y_pred_reg, alpha=0.5)
plt.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'r--')
plt.xlabel('Valores reales (Regresión Lineal)')
plt.ylabel('Valores predichos')
plt.title('Regresión Lineal: Predicciones vs Reales')
plt.grid(True)
plt.show()

# --- RED NEURONAL ---

# Para red neuronal usamos también 'promSaldoPrest3Um' como feature para ejemplo (aunque no es ideal predecir con la variable objetivo dentro)
X_nn = df_features[['frecuencia_transacciones', 'rngEdad', 'promSaldoPrest3Um']].values
y_nn = df_features['promSaldoPrest3Um'].values

# Normalizar variables
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_nn)

# División para red neuronal (80/20)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_nn, test_size=0.2, random_state=42)

# Crear modelo de red neuronal
model_nn = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)  # Salida para regresión
])

model_nn.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Entrenar red neuronal
history = model_nn.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

# Gráfico evolución error entrenamiento
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Pérdida entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida validación')
plt.xlabel('Épocas')
plt.ylabel('Error (MSE)')
plt.title('Evolución del error durante entrenamiento (Red Neuronal)')
plt.legend()
plt.grid(True)
plt.show()

# Predecir con red neuronal
y_pred_nn = model_nn.predict(X_test).flatten()

# Gráfico predichos vs reales red neuronal
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred_nn, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Valores reales (Red Neuronal)')
plt.ylabel('Valores predichos')
plt.title('Red Neuronal: Predicciones vs Reales')
plt.grid(True)
plt.show()

# --- PARTE 5: Lógica de programación ---
# Analizar diferencias absolutas entre predicción y valor real con bucle, condición, lista y diccionario

diferencias = []
categorias_error = {
    "Muy bajo (<1000)": 0,
    "Bajo (1000-5000)": 0,
    "Medio (5000-10000)": 0,
    "Alto (>10000)": 0
}

for i in range(len(y_test)):
    diff = abs(y_test[i] - y_pred_nn[i])
    diferencias.append(diff)
    
    if diff < 1000:
        categorias_error["Muy bajo (<1000)"] += 1
    elif diff < 5000:
        categorias_error["Bajo (1000-5000)"] += 1
    elif diff < 10000:
        categorias_error["Medio (5000-10000)"] += 1
    else:
        categorias_error["Alto (>10000)"] += 1

print("\nClasificación de errores en las predicciones de la red neuronal:")
for categoria, cuenta in categorias_error.items():
    print(f"{categoria}: {cuenta} casos")

# Histograma diferencias
plt.figure(figsize=(8,5))
plt.hist(diferencias, bins=30, color='skyblue', edgecolor='black')
plt.title("Distribución de diferencias absolutas (Red Neuronal)")
plt.xlabel("Diferencia absoluta")
plt.ylabel("Cantidad de muestras")
plt.grid(True)
plt.show()
