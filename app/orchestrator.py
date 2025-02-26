# Archivo: orchestrator.py
import os
import subprocess

# Archivos necesarios
required_files = [
    'product_embeddings.npy',
    'interaction_embeddings.npy',
    'collaborative_similarity_matrix.npy'
]


# Obtiene la ruta absoluta del directorio raíz del proyecto
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Función para verificar existencia de archivos
def check_files(files):
    for file in files:
        if not os.path.exists(os.path.join(project_root, 'data', file)):
            return False
    return True

# Paso 1: Ejecutar data_preprocess.py si no existen los embeddings
if not check_files(['product_embeddings.npy', 'interaction_embeddings.npy']):
    print("Ejecutando data_preprocess.py...")
    subprocess.run(['python3', 'app/data_preprocess.py'], cwd=project_root)
else:
    print("Embeddings ya generados. Saltando data_preprocess.py...")

# Paso 2: Ejecutar model.py si no existe collaborative_similarity_matrix.npy
if not os.path.exists(os.path.join(project_root, 'data', 'collaborative_similarity_matrix.npy')):
    print("Ejecutando model.py...")
    subprocess.run(['python3', 'app/model.py'], cwd=project_root)
else:
    print("Matriz de similitudes colaborativas ya generada. Saltando model.py...")

# Paso 3: Ejecutar FastAPI con main.py
print("Iniciando FastAPI con main.py...")
subprocess.run(['uvicorn', 'app.main:app', '--host', '0.0.0.0', '--port', '8080'], cwd=project_root)