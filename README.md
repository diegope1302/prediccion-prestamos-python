# Predicción de Montos Promedio de Préstamos

## 🎯 Objetivo del Trabajo

Este proyecto tiene como propósito construir modelos de regresión (lineal y red neuronal) para predecir el monto promedio de préstamos de clientes, basándonos en información como edad, frecuencia de transacciones y otros datos relevantes.

---

## 📊 Descripción del Dataset

El conjunto de datos se encuentra en el archivo `dataBasePrestDigital.csv` y contiene información sobre:

- Clientes identificados por un ID único
- Rango de edad (`rngEdad`)
- Promedio de saldo de préstamo en los últimos 3 meses (`promSaldoPrest3Um`)
- Frecuencia de transacciones por cliente (calculada)

Solo se consideran los registros donde el monto promedio de préstamo sea mayor a 0 para asegurar la calidad de los datos.

---

## 🧰 Librerías Utilizadas

- `pandas`
- `matplotlib`
- `scikit-learn`
- `tensorflow` / `keras`

---

## 🧠 Modelos Utilizados

### 🔷 Regresión Lineal

Se utilizó `LinearRegression` de scikit-learn. El modelo fue entrenado con:

- Frecuencia de transacciones
- Edad
- Monto promedio préstamo

Se graficaron los valores reales vs. los valores predichos para visualizar el rendimiento.

### 🔶 Red Neuronal

Se construyó una red neuronal secuencial con Keras:

- Capa densa de 64 neuronas (ReLU)
- Capa densa de 32 neuronas (ReLU)
- Capa de salida con 1 neurona (regresión)

Se entrenó con los mismos datos que la regresión lineal y se graficó la evolución del error (loss), así como la comparación de predicciones vs. valores reales.

---

## 🧪 Lógica de Programación

Se utilizó:

- ✅ Un bucle `for` para recorrer predicciones y crear estructuras dinámicas.
- ✅ Una condición `if` para evaluar predicciones por rangos.
- ✅ Una `lista` para almacenar etiquetas de precisión.
- ✅ Un `diccionario` para agrupar clientes por tramos de edad.

Esto permitió hacer análisis personalizados y mostrar resultados más completos.

---

## 📷 Gráficas

### Regresión Lineal
![Regresión Lineal](Figure_regresion.png)

### Red Neuronal
![Red Neuronal](Figure_red_neuronal.png)

---

## 📌 Conclusiones Personales

- La red neuronal mostró un mejor ajuste en los datos con menor error medio absoluto.
- La regresión lineal es útil para tener un modelo interpretable y rápido de ejecutar.
- La estandarización de variables fue clave para mejorar el rendimiento.
- La integración de visualizaciones permite interpretar los resultados con mayor claridad.
- Este proyecto me permitió reforzar conceptos de análisis de datos, regresión, redes neuronales y flujo de trabajo en GitHub.

---

## 🔗 Enlace al repositorio

👉 [https://github.com/tu_usuario/tu_repositorio](https://github.com/tu_usuario/tu_repositorio)

