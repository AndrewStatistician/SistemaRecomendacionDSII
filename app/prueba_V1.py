# Importar librerías necesarias
import numpy as np
import pandas as pd
from evaluation import dividir_datos, evaluar_modelo
from evaluation import dividir_datos, evaluar_modelo, k_fold_cross_validation

import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(project_root, 'data')

interactions_df = pd.read_csv(os.path.join(data_dir, 'interactions.csv'))


# Carga de los embeddings correctamente usando np.load()
product_embeddings = np.load(os.path.join(data_dir, 'product_embeddings.npy'))
interaction_embeddings = np.load(os.path.join(data_dir, 'interaction_embeddings.npy'))

# Cargar los datos y embeddings generados previamente


# Dividir los datos en entrenamiento y prueba
train_df, test_df = dividir_datos(interactions_df)

# Evaluar el modelo
resultados = evaluar_modelo(test_df, interaction_embeddings, product_embeddings, k=5)

# Imprimir resultados
print("Resultados de la Evaluación del Modelo:")
for metric, score in resultados.items():
    print(f"{metric}: {score:.4f}")
