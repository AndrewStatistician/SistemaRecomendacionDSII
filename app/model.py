# Archivo: model.py

import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import torch

# Obtiene la ruta absoluta del directorio raíz del proyecto
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(project_root, 'data')

# -------------------------
# CARGA DE EMBEDDINGS
# -------------------------

# Cargar embeddings generados en data_preprocess.py
product_embeddings = np.load(os.path.join(data_dir, 'product_embeddings.npy'))
interaction_embeddings = np.load(os.path.join(data_dir, 'interaction_embeddings.npy'))

# Cargar DataFrames originales para índices y metadatos
interactions_df = pd.read_csv(os.path.join(data_dir, 'interactions.csv'))

# -------------------------
# EMBEDDINGS DE INTERACCIONES
# -------------------------

# Agrupar interacciones por usuario y promediar embeddings ponderados por rating
user_embeddings = []
for user_id, group in interactions_df.groupby('user_id'):
    indices = group.index.tolist()
    user_inter_embeddings = interaction_embeddings[indices]
    ratings = torch.tensor(group['rating'].values).float().view(-1, 1)
    weighted_embeddings = torch.tensor(user_inter_embeddings) * ratings
    user_embedding = torch.mean(weighted_embeddings, dim=0)
    user_embeddings.append(user_embedding)

# Convertir a tensor final y luego a numpy
user_embeddings = torch.stack(user_embeddings)
user_embeddings = user_embeddings.cpu().numpy()
print("Dimensión de Embeddings de Usuarios (Interacciones):", user_embeddings.shape)

# -------------------------
# LIMPIEZA DE NaN EN EMBEDDINGS
# -------------------------

# Reemplazar NaN con ceros en los embeddings
user_embeddings = np.nan_to_num(user_embeddings)
product_embeddings = np.nan_to_num(product_embeddings)

# Validar si existen NaN después de la limpieza
print("¿Hay NaN en User Embeddings?", np.isnan(user_embeddings).any())
print("¿Hay NaN en Product Embeddings?", np.isnan(product_embeddings).any())

# Verificar dimensiones finales
print("Dimensión Final de Embeddings de Usuarios:", user_embeddings.shape)
print("Dimensión Final de Embeddings de Productos:", product_embeddings.shape)

# -------------------------
# FILTRADO COLABORATIVO CON EMBEDDINGS
# -------------------------

# Similaridad colaborativa usando embeddings combinados
def batch_cosine_similarity(user_embeddings, product_embeddings, batch_size=50):
    similarities = []
    num_batches = int(np.ceil(user_embeddings.shape[0] / batch_size))
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, user_embeddings.shape[0])
        batch_sim = cosine_similarity(user_embeddings[start_idx:end_idx], product_embeddings)
        similarities.append(batch_sim.astype('float32'))
    return np.vstack(similarities)

print("Calculando similitudes colaborativas en lotes...")
collaborative_similarities = batch_cosine_similarity(user_embeddings, product_embeddings, batch_size=50)
np.save(os.path.join(data_dir, 'collaborative_similarity_matrix.npy'), collaborative_similarities)

print("Cálculo de similitudes colaborativas completado y guardado.")
