# Archivo: main.py
from fastapi import FastAPI, Query
# Archivo: main.py

from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List
import numpy as np
import pandas as pd
import os
import uvicorn

app = FastAPI()

class Recommendation(BaseModel):
    product_id: int
    name: str
    category: str

# Obtiene la ruta absoluta del directorio raíz del proyecto
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(project_root, 'data')

# Carga de Datos usando rutas absolutas
products_df = pd.read_csv(os.path.join(data_dir, 'products.csv'), delimiter=';')
collaborative_similarities = np.load(os.path.join(data_dir, 'collaborative_similarity_matrix.npy'))

# -------------------------
# FUNCION DE RECOMENDACIONES
# -------------------------

# Función para obtener recomendaciones usando Filtrado Colaborativo
async def get_recommendations(user_id, top_n=5):
    user_index = user_id - 1  # Ajuste del índice para que empiece desde 0
    scores = collaborative_similarities[user_index]

    # Obtener los índices de los productos con mayor puntaje
    recommended_product_indices = np.argsort(scores)[::-1][:top_n]

    # Obtener los IDs de productos recomendados
    recommended_product_ids = products_df.iloc[recommended_product_indices]['product_id'].values

    # Hacer join con el DataFrame de productos para obtener name y category
    recommended_products = products_df[products_df['product_id'].isin(recommended_product_ids)][['product_id', 'name', 'category']]
    recommended_products = recommended_products.drop_duplicates().reset_index(drop=True)

    # Convertir el resultado a una lista de diccionarios para la respuesta JSON
    recommendations_list = []
    for _, row in recommended_products.iterrows():
        recommendations_list.append({
            "product_id": row['product_id'],
            "name": row['name'],
            "category": row['category']
        })

    return recommendations_list

# -------------------------
# ENDPOINT DE LA API
# -------------------------

@app.get("/recommendations", response_model=List[Recommendation])
async def get_recommendations_api(user_id: int = Query(..., description="User ID"), top_n: int = Query(5, description="Number of recommendations")):
    return await get_recommendations(user_id, top_n)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
