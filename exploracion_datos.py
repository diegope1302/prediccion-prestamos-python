import pandas as pd

# Cargar el archivo CSV
df = pd.read_csv('dataBasePrestDigital.csv')

# Mostrar las primeras filas
print("Primeras 5 filas:")
print(df.head())

# Mostrar información general del dataset
print("\nInformación general:")
print(df.info())

# Estadísticas descriptivas de las variables numéricas
print("\nEstadísticas descriptivas:")
print(df.describe())

# Verificar si hay valores nulos
print("\nValores nulos por columna:")
print(df.isnull().sum())
