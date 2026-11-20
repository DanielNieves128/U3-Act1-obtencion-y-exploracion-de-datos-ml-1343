import pandas as pd
import os

# URL del dataset público
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"

# Crear carpeta para guardar los datos
os.makedirs("datasets", exist_ok=True)

# Descargar y cargar el CSV
datos = pd.read_csv(url)

# Guardar una copia local
datos.to_csv("datasets/iris.csv", index=False)

# Explorar los datos
print("PRIMERAS 5 FILAS")
print(datos.head())

print("\\nCANTIDAD DE FILAS Y COLUMNAS")
print(datos.shape)

print("\\nNOMBRES DE LAS COLUMNAS")
print(datos.columns.tolist())

print("\\nINFORMACION DEL DATASET")
print(datos.info())

print("\\nCARACTERISTICAS (X)")
X = datos.drop(columns=["species"])
print(X.head())

print("\\nVARIABLE OBJETIVO (y)")
y = datos["species"]
print(y.head())

print("\\nESPECIES REGISTRADAS")
print(y.unique())