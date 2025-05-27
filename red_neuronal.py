import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import re
import numpy as np

# Cargar dataset
df = pd.read_csv('dataBasePrestDigital.csv', sep=';')

# Convertir rngEdad a numérico
def convertir_rango_edad(rango):
    if pd.isna(rango):
        return np.nan
    nums = re.findall(r'\d+', rango)
    if len(nums) == 2:
        return (int(nums[0]) + int(nums[1])) / 2
    elif len(nums) == 1:
        return float(nums[0])
    else:
        return np.nan

df['rngEdad_num'] = df['rngEdad'].apply(convertir_rango_edad)

# Filtrar filas con monto > 0 para tener datos consistentes
df_filtrado = df[df['promSaldoPrest3Um'] > 0]

# Calcular frecuencia de transacciones (total)
frecuencia = df.groupby('cliente').size().reset_index(name='frecuencia_transacciones')

# Promedio saldo préstamo 3 meses para monto objetivo y como feature
prom_saldo = df_filtrado.groupby('cliente')['promSaldoPrest3Um'].mean().reset_index(name='promSaldoPrest3Um')

# Edad numérica media por cliente
edad = df.groupby('cliente')['rngEdad_num'].mean().reset_index(name='rngEdad')

# Unir todas las features en un dataframe
df_features = frecuencia.merge(prom_saldo, on='cliente', how='inner') \
                       .merge(edad, on='cliente', how='inner')

# Variable objetivo (monto promedio)
y = df_features['promSaldoPrest3Um'].values

# Variables predictoras
X = df_features[['frecuencia_transacciones', 'rngEdad', 'promSaldoPrest3Um']].values

# Normalizar (estandarizar) las variables predictoras
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Crear modelo de red neuronal
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)  # Salida para regresión (sin activación)
])

# Compilar modelo
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Entrenar modelo y guardar el historial
history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.2, verbose=1)

# Graficar evolución del error (loss)
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Pérdida entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida validación')
plt.xlabel('Épocas')
plt.ylabel('Error (MSE)')
plt.title('Evolución del error durante el entrenamiento')
plt.legend()
plt.show()

# Predecir sobre datos de prueba
y_pred = model.predict(X_test).flatten()

# Graficar valores predichos vs reales
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Valores reales')
plt.ylabel('Valores predichos')
plt.title('Predicciones vs Valores reales')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.show()
