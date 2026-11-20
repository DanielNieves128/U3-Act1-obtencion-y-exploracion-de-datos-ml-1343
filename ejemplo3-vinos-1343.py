import pandas as pd
import os

# URL del dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

# Crear carpeta
os.makedirs("datasets", exist_ok=True)

# Cargar CSV
# sep=";" indica que los valores están separados por punto y coma
datos = pd.read_csv(url, sep=";")

# Guardar archivo
datos.to_csv("datasets/winequality-red.csv", index=False)

# Explorar datos
print("PRIMERAS 5 MUESTRAS")
print(datos.head())

print("\\nDIMENSIONES")
print(datos.shape)

print("\\nCOLUMNAS")
print(datos.columns.tolist())

print("\\nINFORMACION GENERAL")
print(datos.info())

print("\\nESTADISTICAS")
print(datos.describe())

# Separar características
X = datos.drop(columns=["quality"])

# Variable objetivo
y = datos["quality"]

print("\\nCARACTERISTICAS (X)")
print(X.head())

print("\\nVARIABLE OBJETIVO (y)")
print(y.head())

print("\\nCALIFICACIONES REGISTRADAS")
print(y.value_counts().sort_index())