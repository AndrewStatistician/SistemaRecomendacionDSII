# Archivo: data_preprocess.py

import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from ast import literal_eval

# Obtiene la ruta absoluta del directorio raíz del proyecto
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(project_root, 'data')

# -------------------------
# FUNCIONES AUXILIARES
# -------------------------

# Función para concatenar listas en un solo texto
def concatenar_lista(lista):
    try:
        lista = literal_eval(lista)
        if isinstance(lista, list):
            return ' '.join(lista)
    except (ValueError, SyntaxError):
        pass
    return lista if isinstance(lista, str) else ''

# -------------------------
# CARGA Y DEPURACIÓN DE DATOS
# -------------------------

# Carga de Datos usando rutas absolutas
products_df = pd.read_csv(os.path.join(data_dir, 'products.csv'), delimiter=';')
users_df = pd.read_csv(os.path.join(data_dir, 'users.csv'))
interactions_df = pd.read_csv(os.path.join(data_dir, 'interactions.csv'))

# Rellenar valores nulos
products_df = products_df.fillna(' ')
users_df = users_df.fillna(' ')
interactions_df = interactions_df.fillna(' ')

# Concatenar listas de palabras clave en productos
products_df['palabras_clave'] = products_df['palabras_clave'].apply(concatenar_lista)

# Concatenar toda la información relevante en un solo texto (Productos)
products_df['text'] = (
    products_df['name'] + ". Categoria: " + products_df['category'] + ". " +
    products_df['descripcion'] + ". " + products_df['palabras_clave']
)

# Verificar el resultado de la concatenación en productos
print("Texto Concatenado (Productos):\n", products_df[['text']].head())

# Unir interacciones con productos
user_interactions = pd.merge(interactions_df, products_df, on='product_id')

# Concatenar información relevante de interacciones
user_interactions['text'] = (
    "Interaccion: " + user_interactions['tipo_interaccion'] + ". " +
    "Rating: " + user_interactions['rating'].astype(str) + ". " +
    "Comentario: " + user_interactions['comentario'] + ". " +
    "Descripcion: " + user_interactions['text']
)

# Verificar el resultado de la concatenación en interacciones
print("Texto Concatenado (Interacciones):\n", user_interactions[['text']].head())

# -------------------------
# GENERACIÓN DE EMBEDDINGS
# -------------------------

# Cargar el modelo de SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generación de embeddings de productos
print("Generando embeddings de productos...")
product_embeddings = model.encode(products_df['text'].tolist(), batch_size=16, show_progress_bar=True)
np.save(os.path.join(data_dir, 'product_embeddings.npy'), product_embeddings)

# Generación de embeddings de interacciones
print("Generando embeddings de interacciones...")
interaction_embeddings = model.encode(user_interactions['text'].tolist(), batch_size=16, show_progress_bar=True)
np.save(os.path.join(data_dir, 'interaction_embeddings.npy'), interaction_embeddings)

print("Vectorización de productos e interacciones completada y guardada.")
