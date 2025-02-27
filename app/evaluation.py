# Archivo: evaluation.py

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold

# -------------------------
# DIVISIÓN DE DATOS
# -------------------------

def dividir_datos(interactions_df, test_size=0.2):
    """
    Divide el DataFrame de interacciones en conjunto de entrenamiento y prueba.
    Verifica que la columna 'user_id' esté presente.
    """
    # Verificar existencia de la columna 'user_id'
    if 'user_id' not in interactions_df.columns:
        raise ValueError("La columna 'user_id' no está presente en el DataFrame.")
    print("Columnas en interactions_df antes de dividir:", interactions_df.columns)
    test_indices = interactions_df.groupby('user_id').sample(frac=test_size, random_state=42).index
    train_df = interactions_df.drop(index=test_indices)
    test_df = interactions_df.loc[test_indices]
    return train_df, test_df

# -------------------------
# MÉTRICAS DE EVALUACIÓN
# -------------------------

def precision_at_k(y_true, y_pred, k=5):
    top_k_preds = y_pred[:k]
    relevant_items = set(y_true)
    recommended_items = set(top_k_preds)
    return len(recommended_items & relevant_items) / len(recommended_items)


def recall_at_k(y_true, y_pred, k=5):
    top_k_preds = y_pred[:k]
    relevant_items = set(y_true)
    recommended_items = set(top_k_preds)
    return len(recommended_items & relevant_items) / len(relevant_items)


def ndcg_at_k(y_true, y_pred, k=5):
    """
    Calcula NDCG@K (Normalized Discounted Cumulative Gain).
    Ajustado para trabajar con sklearn.ndcg_score().
    """
    # Crear lista completa de relevancias para todos los productos evaluados
    relevances = np.zeros(len(y_pred))
    relevances[:k] = [1 if item in y_true else 0 for item in y_pred[:k]]

    # Crear lista binaria de relevancias verdaderas
    true_relevances = np.array([1 if item in y_true else 0 for item in y_pred])

    # Calcular NDCG usando listas de igual longitud
    return ndcg_score([true_relevances], [relevances])


def map_at_k(y_true, y_pred, k=5):
    relevant_items = set(y_true)
    precisions = []
    for i, pred in enumerate(y_pred[:k], start=1):
        if pred in relevant_items:
            precisions.append(len(precisions) / i)
    return np.mean(precisions) if precisions else 0


def mrr_at_k(y_true, y_pred, k=5):
    for i, pred in enumerate(y_pred[:k], start=1):
        if pred in y_true:
            return 1 / i
    return 0

# -------------------------
# K-FOLD CROSS VALIDATION
# -------------------------
# -------------------------
# EVALUACIÓN DEL MODELO
# -------------------------

def evaluar_modelo(test_df, user_embeddings, product_embeddings, k=5):
    """
    Evalúa el modelo utilizando embeddings y calcula las métricas.
    """
    resultados = {'precision': [], 'recall': [], 'ndcg': [], 'map': [], 'mrr': []}
    for user_id, group in test_df.groupby('user_id'):
        true_items = group['product_id'].tolist()
        user_idx = user_id - 1
        user_vector = user_embeddings[user_idx].reshape(1, -1)
        similitudes = cosine_similarity(user_vector, product_embeddings)[0]
        recommended_indices = np.argsort(similitudes)[::-1][:k]
        recommended_items = [i+1 for i in recommended_indices]
        
        resultados['precision'].append(precision_at_k(true_items, recommended_items, k))
        resultados['recall'].append(recall_at_k(true_items, recommended_items, k))
        resultados['ndcg'].append(ndcg_at_k(true_items, recommended_items, k))
        resultados['map'].append(map_at_k(true_items, recommended_items, k))
        resultados['mrr'].append(mrr_at_k(true_items, recommended_items, k))

    # Promediar las métricas
    resultados = {metric: np.mean(scores) for metric, scores in resultados.items()}
    return resultados


def k_fold_cross_validation(interactions_df, user_embeddings, product_embeddings, k=5, n_splits=5):
    """
    Realiza K-Fold Cross-Validation y promedia las métricas.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    resultados_cv = {'precision': [], 'recall': [], 'ndcg': [], 'map': [], 'mrr': []}

    for train_index, test_index in kf.split(interactions_df):
        train_df = interactions_df.iloc[train_index]
        test_df = interactions_df.iloc[test_index]

        resultados = evaluar_modelo(test_df, user_embeddings, product_embeddings, k)

        for metric, score in resultados.items():
            resultados_cv[metric].append(score)

    # Promediar métricas a través de los K folds
    resultados_promediados = {metric: np.mean(scores) for metric, scores in resultados_cv.items()}
    return resultados_promediados
