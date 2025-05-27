# 📊 Predicción de Montos de Préstamos con Regresión Lineal y Red Neuronal

## 🎯 Objetivo del Trabajo

Desarrollar un modelo de predicción de montos de préstamos utilizando técnicas de aprendizaje automático, aplicando regresión lineal y redes neuronales en Python. Además, practicar la limpieza, análisis, visualización y entrenamiento de datos reales con lógica de programación y buenas prácticas de documentación.

---

## 📄 Descripción del Dataset

El dataset `dataBasePrestDigital.csv` contiene información transaccional y demográfica de clientes, incluyendo:

- `cliente`: ID único del cliente
- `tipoTx`: tipo de transacción
- `promSaldoPrest3Um`: promedio del saldo del préstamo en los últimos 3 meses
- `rngEdad`: rango de edad (por ejemplo, `<35-45]`)
- `fecha`: fecha de la transacción

> El archivo utiliza el punto y coma (`;`) como separador de columnas.

---

## 🧰 Librerías Utilizadas

- `pandas`: manipulación y limpieza de datos
- `numpy`: operaciones numéricas
- `matplotlib`: visualización de datos
- `scikit-learn`: escalado, regresión lineal, métricas
- `tensorflow.keras`: construcción y entrenamiento de la red neuronal

Instalación:

```bash
pip install pandas matplotlib scikit-learn tensorflow
