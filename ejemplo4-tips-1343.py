import pandas as pd
import os

# URL del dataset
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"

# Crear carpeta
os.makedirs("datasets", exist_ok=True)

# Descargar y cargar CSV
datos = pd.read_csv(url)

# Guardar archivo local
datos.to_csv("datasets/tips.csv", index=False)

# Explorar datos
print("PRIMERAS 5 FILAS")
print(datos.head())

print("\\nDIMENSIONES")
print(datos.shape)

print("\\nNOMBRES DE COLUMNAS")
print(datos.columns.tolist())

print("\\nINFORMACION GENERAL")
print(datos.info())

print("\\nESTADISTICAS")
print(datos.describe())

# Características de entrada
X = datos[["total_bill", "sex", "smoker", "day", "time", "size"]]

# Variable que se desea predecir
y = datos["tip"]

print("\\nCARACTERISTICAS (X)")
print(X.head())

print("\\nVARIABLE OBJETIVO (y)")
print(y.head())

print("\\nPROMEDIO DE PROPINA")
print(y.mean())