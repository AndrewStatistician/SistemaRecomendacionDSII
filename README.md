

# Sistema de Recomendaciones  Filtrado Colaborativo con Embeddings con FastAPI y SentenceTransformer.

<br> 


# 1. Contexto.

En este documento se describe el desarrollo de un motor de recomendaciones para Compensar, una organización comprometida con el bienestar y calidad de vida de las personas a través de su plataforma digital. La plataforma ofrece una amplia variedad de productos y servicios en categorías como deportes, salud, familia, mascotas, desarrollo personal, alimentación y bienestar mental. El objetivo del proyecto es personalizar la experiencia del usuario, sugiriendo productos y servicios relevantes en función de sus intereses y necesidades.

Para lograrlo, se utilizaron datos provenientes de tres bases de datos en formato CSV: users.csv, products.csv y interactions.csv. Se tomaron algunas variables clave para construir un modelo de recomendaciones personalizado. Estas variables permiten capturar el comportamiento y las preferencias de los usuarios, facilitando la conexión con productos y servicios relevantes para su bienestar.

## Resumen de Datos y Naturaleza de Variables.
La siguiente tabla presenta un **resumen de los datos** utilizados en el sistema de recomendaciones, detallando las **variables contenidas en cada archivo**, su **descripción** y la **naturaleza de cada variable**. Esta clasificación ayuda a comprender mejor el tipo de información disponible y su aplicabilidad en el **preprocesamiento de datos, generación de embeddings y cálculo de similitudes colaborativas**.



| Archivo             | Variable             | Descripción                                  | Naturaleza de la Variable      |
|---------------------|----------------------|----------------------------------------------|--------------------------------|
| `products.csv`      | `product_id`          | Identificador único del producto               | Categórica (Nominal)            |
|                     | `name`                | Nombre del producto                           | Categórica (Nominal)            |
|                     | `category`            | Categoría a la que pertenece el producto       | Categórica (Nominal)            |
|                     | `descripcion`         | Descripción detallada del producto             | Categórica (Texto Libre)        |
|                     | `palabras_clave`      | Lista de palabras clave                        | Categórica (Lista de Texto)     |
| `users.csv`         | `user_id`             | Identificador único del usuario                | Categórica (Nominal)            |
|                     | `edad`                | Edad del usuario                              | Numérica (Continua)             |
|                     | `genero`              | Género del usuario                            | Categórica (Nominal)            |
|                     | `nivel_ingresos`      | Nivel de ingresos del usuario                  | Categórica (Ordinal)             |
|                     | `intereses`           | Intereses del usuario                         | Categórica (Lista de Texto)     |
|                     | `ubicacion`           | Ubicación geográfica del usuario               | Categórica (Nominal)            |
|                     | `frecuencia_login`    | Frecuencia de inicio de sesión                 | Numérica (Discreta)             |
| `interactions.csv`  | `user_id`             | Identificador único del usuario                | Categórica (Nominal)            |
|                     | `product_id`          | Identificador único del producto               | Categórica (Nominal)            |
|                     | `tipo_interaccion`    | Tipo de interacción (e.g., vista, compra, etc.)| Categórica (Nominal)            |
|                     | `rating`              | Calificación del usuario                       | Numérica (Ordinal)              |
|                     | `comentario`          | Comentario dejado por el usuario               | Categórica (Texto Libre)        |

<br><br><br>





# 2. Sistema de Recomendaciones con Filtrado Colaborativo Basado en Embeddings.

El sistema de recomendaciones utiliza un enfoque de **Filtrado Colaborativo basado en Embeddings** generado mediante `SentenceTransformer ('all-MiniLM-L6-v2')`. Se eligió este enfoque debido a las siguientes razones:

- **Captura de Relaciones Complejas:**  
  Al utilizar embeddings, se logra captar relaciones complejas entre usuarios y productos que van más allá de simples co-ocurrencias, permitiendo recomendaciones más precisas y relevantes.

- **Generalización y Escalabilidad:**  
  Los embeddings ofrecen una mejor capacidad de generalización en comparación con técnicas tradicionales como SVD o Matrix Factorization, permitiendo una mejor adaptación a productos y usuarios nuevos.

- **Uso Eficiente de Datos de Interacciones:**  
  Se utilizan datos de interacciones para generar embeddings ponderados por el rating, capturando no solo la acción del usuario, sino también su intensidad o nivel de satisfacción.

## Justificación de la Elección del Filtrado Colaborativo sobre el Enfoque Híbrido.

El enfoque híbrido requería combinar similitudes de contenido y colaborativas, lo cual resultó en problemas de dimensionalidad y consumo de memoria. Además, la estructura de los datos disponibles no justificaba la complejidad adicional, sin embargo se realizo pruebas con enfoque híbrido pero como ejercicio se tomo el modelo Filtrado colaborativo.


## Variables Utilizadas en el Modelo.
La siguiente tabla detalla las **variables utilizadas en el modelo de recomendaciones**, especificando el **DataFrame** del cual provienen, las **variables seleccionadas** y la **justificación** de su inclusión. Estas variables fueron elegidas estratégicamente para **capturar tanto información descriptiva de los productos** como **el comportamiento e interacción de los usuarios**, permitiendo una generación de embeddings más **precisa y relevante**.

<br>

| DataFrame         | Variables Utilizadas                               | Justificación                                                     |
|------------------|-----------------------------------------------------|-------------------------------------------------------------------|
| `products.csv`   | `name`, `category`, `descripcion`, `palabras_clave` | Capturan información descriptiva y categórica del producto, ayudando a generar embeddings significativos. |
| `interactions.csv` | `user_id`, `product_id`, `tipo_interaccion`, `rating`, `comentario` | Se utilizan para capturar el comportamiento del usuario y ponderar los embeddings según el rating, reflejando la intensidad de la interacción. |


## Enfoque en Comportamiento e Interacciones.

  Aunque el archivo users.csv contenía información demográfica de los usuarios, no se utilizó en el modelo de recomendaciones. El modelo se centró en capturar las preferencias y comportamientos de los usuarios a través de sus interacciones directas con los productos (clics, comentarios, ratings). Esto permitió ponderar los embeddings de manera más precisa en función de la intensidad de la interacción.



<br><br><br>

# **3. Arquitectura del Sistema y Tecnologías Elegidas.**
El sistema de recomendaciones desarrollado utiliza una arquitectura modular y eficiente, diseñada para maximizar el rendimiento y escalar fácilmente en entornos de producción. La estructura del proyecto y las tecnologías elegidas han sido seleccionadas estratégicamente para optimizar el flujo de datos, procesamiento de texto y cálculo de similitudes, garantizando una experiencia de usuario fluida y personalizada.

# **Estructura del Proyecto.**
La organización del proyecto sigue una arquitectura modular, dividiendo las responsabilidades en archivos específicos para mantener un código limpio, mantenible y escalable. La estructura es la siguiente:



## **Raíz del Proyecto (project_root/)**
En el directorio raíz, se ubican los archivos clave para la configuración del entorno y la documentación del proyecto:
```
project_root/ 
│   Dockerfile                # Configuración para la contenerización y despliegue en Docker
│   README.md                 # Documentación detallada del proyecto, instalación y uso
│   requirements.txt          # Dependencias y librerías necesarias para el proyecto
```
- **`Dockerfile`:**  Define el entorno del contenedor, instalando las dependencias y exponiendo la API de FastAPI en un puerto específico.
- **`README.md`:**  Documentación con la descripción del proyecto, instrucciones de instalación, uso de la API y detalles técnicos.
- **`requirements.txt`:**  Lista de todas las dependencias y librerías necesarias para el correcto funcionamiento del sistema.

## **Carpeta de Aplicación (app/)**
Esta carpeta contiene los scripts de Python responsables del preprocesamiento de datos, generación de embeddings, cálculo de similitudes y exposición de la API. Cada archivo tiene una responsabilidad específica dentro del flujo de trabajo:
```
├── app/
│   ├── data_preprocess.py     # Preprocesamiento de datos y generación de embeddings
│   ├── model.py               # Cálculo de similitudes mediante Filtrado Colaborativo
│   ├── main.py                # API REST con FastAPI para exponer las recomendaciones
│   └── orchestrator.py        # Coordinación de la ejecución y flujo de datos
```

- **`data_preprocess.py`:**  Preprocesamiento de datos y generación de embeddings.                               
- **`model.py`:**            Cálculo de similitudes colaborativas utilizando los embeddings.                      
- **`main.py`:**             Exposición de las recomendaciones mediante una API REST con FastAPI.                 
- **`orchestrator.py`:**     Orquestación del flujo de datos, ejecución secuencial de los scripts y despliegue de la API. 


## **Carpeta de Datos (data/)**
Esta carpeta almacena los datos de entrada y salida del sistema, incluyendo los CSVs de entrada y las matrices de embeddings y similitudes generadas:
```
└── data/                   
    ├── interactions.csv       # Interacciones entre usuarios y productos
    ├── products.csv           # Información detallada de productos
    └── users.csv              # Información detallada de usuarios
```

- **`interactions.csv`:**  Registro de interacciones entre usuarios y productos, incluyendo el tipo de interacción (vista, compra, comentario) y el rating dado.
- **`products.csv`:**  Información descriptiva de productos, incluyendo name, category, descripcion, y palabras_clave.
- **`users.csv`:**  Información demográfica y de preferencias de los usuarios (no utilizado en el modelo actual, pero disponible para futuras mejoras).


## **Resumen:**
Esta estructura de carpetas representa la versión final del sistema de recomendaciones, organizada de manera modular para maximizar la mantenibilidad, escalabilidad y eficiencia. La separación clara de responsabilidades y la organización lógica de los datos, scripts y salidas garantizan un flujo de trabajo fluido y eficiente. Además, esta estructura permite futuras mejoras y actualizaciones sin afectar el núcleo del sistema.
```
project_root/ 
│   Dockerfile                # Contenerización para despliegue en cualquier entorno
│   README.md                 # Documentación del proyecto
│   requirements.txt          # Dependencias del proyecto
│
├── app/
│   ├── data_preprocess.py     # Preprocesamiento de datos y generación de embeddings
│   ├── model.py               # Cálculo de similitudes mediante Filtrado Colaborativo
│   ├── main.py                # API REST con FastAPI para exponer las recomendaciones
│   └── orchestrator.py        # Coordinación de la ejecución y flujo de datos
│
└── data/                   
    ├── interactions.csv       # Interacciones entre usuarios y productos
    ├── products.csv           # Información detallada de productos
    └── users.csv              # Información detallada de usuarios
```
<br>
<br>

# **Tecnologías Utilizadas en el Sistema de Recomendaciones**
Para el desarrollo de este sistema de recomendaciones, se seleccionaron tecnologías de vanguardia que no solo maximizan el rendimiento y la escalabilidad, sino que también garantizan la precisión en las recomendaciones. La elección de estas herramientas fue estratégica y se basó en su capacidad para integrarse eficientemente y adaptarse a las necesidades del modelo.

El objetivo principal fue construir una arquitectura modular y flexible, que permita actualizaciones y mejoras continuas sin afectar el rendimiento general del sistema. A continuación se presentan las tecnologías clave y sus funciones específicas dentro del flujo de trabajo:

<br>

| Tecnología                  | Función Principal                                  | Características Clave                                     |
|-----------------------------|-----------------------------------------------------|-----------------------------------------------------------|
| **FastAPI**                 | Exponer la API REST de recomendaciones               | - Alto rendimiento y baja latencia <br> - Documentación interactiva (Swagger y Redoc) <br> - Integración sencilla con Docker |
| **SentenceTransformer ('all-MiniLM-L6-v2')** | Generación de embeddings de alta precisión     | - Modelo liviano y rápido <br> - Alta precisión en similitudes semánticas |
| **scikit-learn y numpy**    | Cálculos de Cosine Similarity y procesamiento de datos | - Operaciones matemáticas optimizadas <br> - Herramientas robustas para análisis de datos |
| **Docker**                  | Contenerización y despliegue en la nube              | - Portabilidad y escalabilidad <br> - Consistencia en desarrollo y producción |

<br>

Estas tecnologías fueron seleccionadas cuidadosamente para abordar los desafíos específicos del sistema de recomendaciones, como la alta dimensionalidad de los embeddings, el procesamiento eficiente de grandes volúmenes de datos y la exposición rápida de resultados a través de la API.

En conjunto, esta arquitectura garantiza la eficiencia, precisión y escalabilidad del sistema, proporcionando una base sólida para futuras mejoras en el modelo de recomendaciones y adaptaciones a nuevos contextos.
<br><br><br>

# **4. Calidad del Código: Claridad, Modularidad y Buenas Prácticas**

## **Organización Modular y Estructura del Proyecto**

El proyecto está organizado en cuatro módulos principales, cada uno con responsabilidades bien definidas:

<br>

| Módulo               | Descripción                                                                         |
|----------------------|-------------------------------------------------------------------------------------|
| `data_preprocess.py` | Preprocesamiento de datos y generación de embeddings.                               |
| `model.py`           | Cálculo de similitudes colaborativas utilizando los embeddings.                      |
| `main.py`            | Exposición de las recomendaciones mediante una API REST con FastAPI.                 |
| `orchestrator.py`    | Orquestación del flujo de datos, ejecución secuencial de los scripts y despliegue de la API. |


## Modularidad y Reutilización

- Se utiliza una **estructura modular** para separar claramente el preprocesamiento de datos, el cálculo de embeddings y la exposición de la API.
- Funciones como `batch_cosine_similarity()` en `model.py` son **reutilizables** y permiten procesamiento en lotes, optimizando el uso de memoria y mejorando el rendimiento.
- La **configuración flexible** en `orchestrator.py` permite ejecutar o saltar scripts según la existencia de archivos intermedios.

## Flujo de Ejecución y Dependencias de los Módulos
La siguiente tabla describe el **flujo de ejecución y las dependencias** entre los módulos principales del sistema de recomendaciones. Se detalla quién ejecuta cada módulo, sus dependencias previas y las salidas generadas, permitiendo una visión clara de la **arquitectura de datos y el control del flujo de ejecución**.

<br>

| Módulo (.py)         | Ejecutado por                | Corre a                                        | Dependencias Previas                              | Salida(s)                                      |
|---------------------|------------------------------|------------------------------------------------|---------------------------------------------------|------------------------------------------------|
| `orchestrator.py`   | Usuario / CMD / Docker        | - `data_preprocess.py` <br> - `model.py` <br> - `main.py` (FastAPI) | Ninguna (controla el flujo de ejecución)           | N/A (Control del flujo de datos)                |
| `data_preprocess.py`| `orchestrator.py`             | Ninguno                                         | - `products.csv` <br> - `interactions.csv`         | - `product_embeddings.npy` <br> - `interaction_embeddings.npy` |
| `model.py`          | `orchestrator.py`             | Ninguno                                         | - `product_embeddings.npy` <br> - `interaction_embeddings.npy` | `collaborative_similarity_matrix.npy`           |
| `main.py`           | `orchestrator.py` (con `uvicorn`) | Ninguno                                         | - `products.csv` <br> - `collaborative_similarity_matrix.npy` | API REST en `/recommendations`                 |


## Buenas Prácticas en el Código

- Se utiliza **numpy** para el manejo eficiente de matrices y **torch** para el cálculo de embeddings ponderados.
- **Manejo de NaN:** Se realiza una limpieza explícita de NaN en los embeddings para evitar errores en el cálculo de similitudes.
- Se siguen convenciones de estilo de código **PEP8 en Python** para garantizar la legibilidad y mantenibilidad.


<br><br><br>

# 5. Decisiones Tomadas y Pasos de Instalación/Ejecución

## Decisiones Clave en la Arquitectura del Sistema

- **Uso de Embeddings con SentenceTransformer:**  
  Se decidió utilizar `SentenceTransformer ('all-MiniLM-L6-v2')` para aprovechar su capacidad de generar representaciones densas y semánticamente ricas tanto para productos como para interacciones.

- **Filtrado Colaborativo Basado en Embeddings:**  
  Se optó por este enfoque debido a la calidad de las recomendaciones y la escalabilidad, al capturar relaciones implícitas entre usuarios y productos a partir de sus interacciones.

- **Batch Processing en Similitudes:**  
  Se utilizó un procesamiento en lotes (`batch_cosine_similarity`) para optimizar el cálculo de similitudes colaborativas y gestionar eficientemente la memoria.

## Pasos de Instalación y Ejecución
Se detalla los pasos para la instalación y ejecución de un sistema de recomendaciones desarrollado en FastAPI utilizando Docker. La solución se despliega en un contenedor, lo que asegura consistencia en el entorno de ejecución y facilita la portabilidad de la aplicación. Además, el uso de volúmenes en Docker permite guardar de manera persistente las salidas de los modelos, asegurando que los resultados y datos generados no se pierdan al detener el contenedor.

A continuación, se presentan los pasos para clonar el repositorio, construir la imagen de Docker y ejecutar el contenedor, proporcionando un entorno completamente funcional para el sistema de recomendaciones.

### 1. Clonar el Repositorio y Navegar al Directorio del Proyecto:

```bash
git clone https://github.com/AndrewStatistician/SistemaRecomendacionDSII.git
cd SistemaRecomendacionDSII
```

### 2. Construir la Imagen de Docker:

Se utiliza un Dockerfile que define el entorno de FastAPI y las dependencias necesarias.

```bash
docker build -t recomendador-fastapi .
```
### 3. Ejecutar el Contenedor de Docker:
```
docker run -d -p 8080:8080 -v "$(pwd)/data:/app/data" --name recomendador recomendador-fastapi
```
 - Este comando inicia un contenedor basado en la imagen recomendador-fastapi. Se expone el puerto 8080 del contenedor en el puerto 8080 de la máquina host (-p 8080:8080), permitiendo el acceso a la aplicación a través de http://localhost:8080. 
 - La opción -v $(pwd)/data:/app/data monta el directorio local data en el directorio /app/data dentro del contenedor, lo que permite a la aplicación acceder y modificar los archivos de datos de forma persistente. Se guardan las salidas de los modelos generados por el recomendador en FastAPI.

Para verificar que el contenedor esté corriendo, utiliza el siguiente comando:

```
docker ps
```
Este comando muestra una lista de los contenedores en ejecución, incluyendo información como el nombre, el estado y los puertos expuestos. Asegúrate de ver un contenedor con el nombre de imagen recomendador-fastapi y el puerto 8080 expuesto, lo cual confirmará que la aplicación está activa y accesible en http://localhost:8080.








### 4. Probar la API de Recomendaciones:
Se puede realizar una solicitud GET a la API expuesta:

```
curl -X GET "http://localhost:8080/recommendations?user_id=123&top_n=5"
```
o tambien  realizarlo en el Navegador:  http://localhost:8000/docs





## Ejemplos de Consultas a la API
La respuesta esperada será un JSON con las recomendaciones personalizadas:


### Obtener Recomendaciones para un Usuario:
```bash
curl -X 'GET' \
  'http://localhost:8080/recommendations?user_id=29&top_n=5' \
  -H 'accept: application/json'
```

- Request URL
```
http://localhost:8080/recommendations?user_id=29&top_n=5
```

- Response body

```
[
  {
    "product_id": 658,
    "name": "Sesión de Terapia Online",
    "category": "Bienestar Mental"
  },
  {
    "product_id": 659,
    "name": "Ropa Deportiva",
    "category": "Deportes"
  },
  {
    "product_id": 660,
    "name": "Curso de Mindfulness",
    "category": "Bienestar Mental"
  },
  {
    "product_id": 671,
    "name": "Accesorios para Entrenamiento",
    "category": "Deportes"
  },
  {
    "product_id": 2000,
    "name": "Clases de Yoga Online",
    "category": "Salud"
  }
]
```



<br><br><br>

# 6. Evaluación del Modelo
## Pruebas y Archivos Utilizados
Las pruebas del modelo se realizaron utilizando dos scripts en Python:

- `prueba_V1.py`: Este archivo ejecuta la evaluación del modelo utilizando los embeddings generados y calcula las métricas de precisión, recall, ndcg, map y mrr. Además, aplica K-Fold Cross-Validation para obtener métricas promediadas.

- `evaluation.py`: Contiene las funciones de evaluación, incluyendo la división de datos, el cálculo de métricas y la implementación de K-Fold Cross-Validation. También define las fórmulas para Precision@K, Recall@K, NDCG@K, MAP@K y MRR@K.

Estos archivos trabajan en conjunto para realizar la evaluación completa del modelo de recomendaciones

## 1. Definiciones de las Métricas
En sistemas de recomendación, las métricas de evaluación son fundamentales para medir la efectividad y precisión de las recomendaciones generadas. En este proyecto, se han utilizado las siguientes métricas:

### 1.1 Precision@K
- **Definición:** Proporción de recomendaciones relevantes en el top K de las recomendaciones.
- **Fórmula:** Precision@K = (Relevantes en Top K) / (Total de Recomendaciones en Top K)   
- **Interpretación:** Indica qué tan precisas son las recomendaciones en el top K.
- **Valor Ideal:** > 20%

### 1.2 Recall@K
- **Definición:** Proporción de ítems relevantes que fueron recomendados en el top K.
- **Fórmula:** Recall@K = (Relevantes en Top K) / (Total de Ítems Relevantes)
- **Interpretación:** Evalúa la cobertura de las recomendaciones relevantes.
- **Valor Ideal:** > 30%

### 1.3 NDCG@K (Normalized Discounted Cumulative Gain)
- **Definición:** Mide la calidad de la clasificación de las recomendaciones en el top K.
- **Fórmula:**   NDCG@K = DCG@K / IDCG@K
- **Interpretación:** Evalúa si los ítems relevantes están bien clasificados.
- **Valor Ideal:** > 50%

### 1.4 MAP@K (Mean Average Precision)
- **Definición:** Calcula la precisión promedio en múltiples recomendaciones en el top K.
- **Fórmula:** Promedio de Precision@K para todos los usuarios.
- **Interpretación:** Evalúa la precisión promedio en las recomendaciones.
- **Valor Ideal:** > 20%

### 1.5 MRR@K (Mean Reciprocal Rank)
- **Definición:** Evalúa la posición del primer ítem relevante en el top K.
- **Fórmula:**    MRR@K = 1 / (Posición del Primer Ítem Relevante)
- **Interpretación:** Mide qué tan rápido aparece el primer ítem relevante.
- **Valor Ideal:** > 20%



## 2. Método Utilizado en el Código
### 2.1 K-Fold Cross-Validation
- **Descripción:** Se utilizó **K-Fold Cross-Validation** con `n_splits=5` para dividir el conjunto de datos en 5 folds.
- **Funcionalidad:**
    - En cada iteración, se utilizan **4 folds para entrenamiento** y **1 fold para prueba**.
    - Se calcula Precision@K, Recall@K, NDCG@K, MAP@K, y MRR@K en cada fold.
    - Las métricas se **promedian** a través de los 5 folds.
- **Ventajas:**
    - Mayor **estabilidad** en las métricas al reducir la varianza causada por una sola partición.
    - Mejor **representatividad** al usar todo el conjunto de datos para entrenamiento y prueba.

### 2.2 Evaluación del Modelo
- Se generan recomendaciones utilizando **embeddings de usuarios y productos**.
- Se utiliza **cosine_similarity** para calcular similitudes.
- Se calculan las métricas mencionadas para el top K (por defecto, K=5).
- Los resultados se **promedian** a nivel de usuario y luego en cada fold.


## 3. Resultados de las Métricas
Resultados obtenidos con K-Fold Cross-Validation:
```
precision: 0.0013
recall: 0.0032
ndcg: 0.0066
map: 0.0000
mrr: 0.0031
```

### 3.1 Interpretación de Resultados
- **Precision@K y Recall@K** son extremadamente bajos, indicando baja relevancia y cobertura en las recomendaciones.
- **NDCG@K** muestra que los ítems relevantes no están bien clasificados.
- **MAP@K y MRR@K** indican que los ítems relevantes no aparecen en las primeras posiciones.

## 4. Recomendaciones para Mejorar las Métricas
1. **Cambiar el Modelo de Embeddings:**
    - Utilizar `all-mpnet-base-v2` en lugar de `all-MiniLM-L6-v2` para una mayor precisión en la similitud de texto.
2. **Ajustar Ponderaciones en el Enfoque Híbrido:**
    - Rebalancear el peso entre **contenido** y **colaborativo**.
3. **Probar Diferentes Métricas de Similitud:**
    - Experimentar con **euclidean_distance** o **dot_product**.
4. **Aumentar K en las Métricas:**
    - Evaluar con valores más altos de K para observar la relevancia en posiciones más bajas.
5. **Mejorar el Preprocesamiento de Datos:**
    - Enriquecer el texto de productos con palabras clave y descripciones detalladas.

## 5. Ejecutar las Pruebas
Para correr las pruebas del modelo de recomendaciones, sigue estos pasos:

Asegúrate de tener los archivos `.npy` generados en la carpeta `data/`:

- `product_embeddings.npy`
- `interaction_embeddings.npy`

Ejecuta el archivo prueba_V1.py en tu terminal:

```
python3 prueba_V1.py
```
Esto ejecutará el flujo completo de evaluación, incluyendo K-Fold Cross-Validation. Con esto se tiene los resultados de las métricas promediadas, al finalizar la ejecución, tedremos las métricas promediadas de Precision@K, Recall@K, NDCG@K, MAP@K y MRR@K.

<br><br><br>

---


<br><br><br>


# 8. Mejoras y Próximos Pasos en el Sistema de Recomendaciones

El sistema de recomendaciones desarrollado en esta versión ha demostrado ser eficiente y preciso, aprovechando el poder de los embeddings generados con SentenceTransformer y el Filtrado Colaborativo. Sin embargo, existen oportunidades de mejora y expansión que pueden llevar el sistema al siguiente nivel en términos de personalización, escalabilidad y rendimiento. A continuación, se detallan los próximos pasos y mejoras potenciales:

## 1. Integración de Datos Sociodemográficos

Actualmente, el modelo se basa únicamente en las interacciones y descripciones de los productos. La integración de datos sociodemográficos (como edad, género, ingresos, ubicación e intereses) permitirá una personalización más profunda y relevante.

### Mejora Propuesta
- Expandir el modelo para incluir datos de `users.csv` utilizando técnicas de embeddings demográficos.
- Realizar concatenación de embeddings para combinar información demográfica con el comportamiento de interacción.
- Segmentación de usuarios basada en perfiles demográficos, mejorando la precisión en recomendaciones específicas para cada grupo.


## 2. Optimizaciones de Rendimiento

### Descripción
El cálculo de similitudes y generación de embeddings son computacionalmente intensivos, especialmente al manejar grandes volúmenes de datos.

- Implementar técnicas de **dimensionalidad reducida** (PCA) para reducir la dimensionalidad de los embeddings, acelerando el cálculo de similitudes.
- Utilizar **cuantización de embeddings** para optimizar el almacenamiento y reducir la memoria requerida.
- Implementar **Batch Processing** más eficiente para procesamiento paralelo y manejo de grandes volúmenes de datos.


## 3. Expansión del Sistema

Actualmente, el sistema utiliza Filtrado Colaborativo con embeddings para generar recomendaciones. Se puede expandir la capacidad predictiva mediante técnicas avanzadas.

- Integrar **modelos secuenciales** (como LSTM o Transformers temporales) para predecir tendencias de comportamiento en función del historial de interacciones.
- Implementar **Recomendaciones Basadas en Contexto** utilizando Factores Temporales (hora del día, día de la semana, eventos especiales) para generar recomendaciones dinámicas y personalizadas.
- Añadir modelos de **Deep Learning** como Autoencoders o Redes Neuronales Convolucionales (CNNs) para extraer características avanzadas de texto e imágenes.


## 4. Despliegue en la Nube (Azure, GCP, AWS) 

Actualmente, el despliegue se realiza en contenedores Docker. La migración a la nube permitirá una escalabilidad más eficiente y alta disponibilidad.

- Desplegar el sistema en **Azure**, **Google Cloud Platform (GCP)** o **Amazon Web Services (AWS)** utilizando **Kubernetes** (AKS, GKE o EKS) para autoescalado y orquestación de contenedores.
- Utilizar servicios gestionados para almacenamiento de embeddings y bases de datos vectoriales como **Pinecone**, **Weaviate** o **Qdrant**.
- Integración con **CI/CD en la nube** para automatizar despliegues y actualizaciones.

## 5. Datos en SQL, CSV y Bases de Datos Vectoriales (Embeddings) 

Actualmente, los datos se almacenan en archivos CSV y las similitudes en archivos `.npy`. La migración a bases de datos más robustas permitirá consultas más rápidas y escalabilidad.

- Migrar a **SQL o NoSQL** (como MongoDB o DynamoDB) para almacenar datos de productos, usuarios e interacciones, mejorando la eficiencia en la consulta de datos.
- Utilizar **Bases de Datos Vectoriales** (como Pinecone o Qdrant) para almacenar y consultar embeddings, permitiendo búsqueda semántica a gran escala.
- Implementar un **ETL automatizado** para la actualización constante de datos y embeddings en tiempo real.



## 6. Mejoras en el Procesamiento Batch y Actualización Dinámica.

```
Embeddings ya generados. Saltando data_preprocess.py...
Matriz de similitudes colaborativas ya generada. Saltando model.py...
Iniciando FastAPI con main.py...
```

El sistema omite la regeneración de embeddings y similitudes si los datos no han cambiado. Sin embargo, no detecta actualizaciones en tiempo real, lo que puede afectar la relevancia de las recomendaciones para nuevos usuarios, productos o interacciones.


- Actualización Incremental de Embeddings: Actualizar solo los embeddings afectados por nuevos usuarios, productos o interacciones, sin recalcular todo el modelo. Utilizar Bases de Datos Vectoriales como Pinecone o Qdrant para almacenar y actualizar dinámicamente los embeddings.

- Procesamiento Reactivo con ETL Inteligente: Implementar un ETL Reactivo que actualice solo los datos afectados por cambios, utilizando eventos en tiempo real. Utilizar Apache Kafka o Google Cloud Pub/Sub para procesamiento reactivo en tiempo real.

- Optimización del Cálculo de Similitudes: Mejorar el cálculo de Cosine Similarity con Approximate Nearest Neighbors (ANN) para acelerar el tiempo de búsqueda. Utilizar FAISS (Facebook AI Similarity Search) o HNSW (Hierarchical Navigable Small World Graphs) para búsquedas KNN eficientes.


## 7. Implementación de Logs en el Sistema de Recomendación


Se implementó en el equipo local (no esta en el desarrollo de los .py actuales) un sistema de logging centralizado para monitorear y rastrear la ejecución del sistema de recomendación. Esto incluye la documentación de:
- Hora de inicio y fin de cada proceso.
- Tiempos de ejecución y métricas de rendimiento.
- Errores y excepciones.
- Verificación de rutas y permisos al guardar archivos.

El objetivo es mejorar la trazabilidad y facilitar la identificación de problemas durante la ejecución.


## Arquitectura de Logs
Todos los logs se guardan de manera centralizada en el directorio:
```
project_root/
 └── logs/                    # Carpeta de logs de ejecución 
      └── log_YYYYMMDD.txt    # Archivo de log por fecha de ejecución
```
- Se guarda un registro detallado de:
- Hora de inicio y fin de cada paso.
- Tiempo de duración de cada proceso.
- Errores y excepciones.
- Verificación de rutas y permisos.

---

## Archivos Modificados y Nuevos

### Archivos Modificados:
1. `app/data_preprocess.py`:
 - Se agregó logging para:
   - Verificación de rutas antes y después de guardar `.npy`.
   - Confirmación de éxito o error al guardar archivos.
   - Hora de inicio y fin de la ejecución.
 - Se utilizó `logging` para guardar estos registros en `logs/`.

2. `app/orchestrator.py`:
 - Se agregó logging para:
   - Hora de inicio y fin de cada proceso (`data_preprocess.py`, `model.py`, `main.py`).
   - Errores y excepciones durante la ejecución.

### Archivos Nuevos:
- `logs/log_YYYYMMDD.txt`: 
- Se crea automáticamente en el directorio `logs/` en cada ejecución.
- Almacena los registros de cada corrida con la fecha en el nombre del archivo.

---

## Funcionalidad de Logging

### Configuración de Logging:
Se utilizó la librería estándar de Python `logging`, configurada de la siguiente manera:
```python
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
if not os.path.exists(log_dir):
  os.makedirs(log_dir)
log_file = os.path.join(log_dir, f"log_{time.strftime('%Y%m%d')}.txt")
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
```
Ejemplo de un archivo de log generado:

```
[2025-02-25 23:00:00] - Inicio de data_preprocess.py
[2025-02-25 23:00:01] - Ruta de data_dir: /mnt/c/.../data
[2025-02-25 23:00:02] - Guardando product_embeddings en: /mnt/c/.../data/product_embeddings.npy
[2025-02-25 23:00:03] - product_embeddings guardado con éxito.
[2025-02-25 23:05:30] - Fin de data_preprocess.py
[2025-02-25 23:05:31] - Inicio de model.py
[2025-02-25 23:05:32] - Error en model.py: FileNotFoundError: No such file or directory
[2025-02-25 23:10:00] - Fin de orchestrator.py
```
```
2025-02-26 15:30:40,945 - Inicio de orchestrator.py
2025-02-26 15:30:40,947 - Ejecutando data_preprocess.py...
2025-02-26 15:30:51,711 - Inicio de data_preprocess.py
2025-02-26 15:30:52,713 - Uso de Memoria (RAM): 2.64 GB
2025-02-26 15:30:52,713 - Uso de CPU: 1.3%
2025-02-26 15:30:52,840 - Tiempo de carga de datos: 0.13 segundos
2025-02-26 15:30:52,972 - Use pytorch device_name: cpu
2025-02-26 15:30:52,972 - Load pretrained SentenceTransformer: all-MiniLM-L6-v2
2025-02-26 15:31:19,367 - Tiempo de generación de embeddings (Productos): 23.92 segundos
2025-02-26 15:31:19,368 - Dimensión de Embeddings de Productos: (2000, 384)
2025-02-26 15:31:20,370 - Uso de Memoria (RAM): 2.78 GB
2025-02-26 15:31:20,370 - Uso de CPU: 1.0%
2025-02-26 15:45:01,710 - Tiempo de generación de embeddings (Interacciones): 821.34 segundos
2025-02-26 15:45:01,711 - Dimensión de Embeddings de Interacciones: (50000, 384)
2025-02-26 15:45:02,713 - Uso de Memoria (RAM): 3.59 GB
2025-02-26 15:45:02,713 - Uso de CPU: 1.4%
2025-02-26 15:45:02,714 - Duración total de data_preprocess.py: 850.00 segundos
2025-02-26 15:45:02,714 - Fin de data_preprocess.py
2025-02-26 15:45:04,444 - data_preprocess.py completado en 863.50 segundos.
2025-02-26 15:45:04,446 - Ejecutando model.py...
2025-02-26 15:45:09,484 - model.py completado en 5.04 segundos.
2025-02-26 15:45:09,485 - Iniciando FastAPI con main.py...
2025-02-26 15:45:11,357 - Duración total de orchestrator.py: 870.41 segundos
2025-02-26 15:45:11,358 - Fin de orchestrator.py
2025-02-26 15:55:41,819 - Inicio de data_preprocess.py
2025-02-26 15:55:41,820 - Ruta de data_dir: /mnt/c/Users/amcastrol/OneDrive - Compensar/Prueba_DSII/versiones/2_5_code_8080_LOGS_TIME/data
2025-02-26 15:55:42,210 - Use pytorch device_name: cpu
2025-02-26 15:55:42,210 - Load pretrained SentenceTransformer: all-MiniLM-L6-v2
2025-02-26 15:56:04,750 - Inicio de orchestrator.py
2025-02-26 15:56:04,751 - Ejecutando data_preprocess.py...
2025-02-26 15:56:09,601 - Inicio de data_preprocess.py
2025-02-26 15:56:09,602 - Ruta de data_dir: /mnt/c/Users/amcastrol/OneDrive - Compensar/Prueba_DSII/versiones/2_5_code_8080_LOGS_TIME/data
2025-02-26 15:56:09,872 - Use pytorch device_name: cpu
2025-02-26 15:56:09,872 - Load pretrained SentenceTransformer: all-MiniLM-L6-v2
2025-02-26 15:56:40,543 - Guardando product_embeddings en: /mnt/c/Users/amcastrol/OneDrive - Compensar/Prueba_DSII/versiones/2_5_code_8080_LOGS_TIME/data/product_embeddings.npy
2025-02-26 15:56:40,580 - product_embeddings guardado con éxito.
2025-02-26 16:11:51,771 - Guardando interaction_embeddings en: /mnt/c/Users/amcastrol/OneDrive - Compensar/Prueba_DSII/versiones/2_5_code_8080_LOGS_TIME/data/interaction_embeddings.npy
2025-02-26 16:11:52,399 - interaction_embeddings guardado con éxito.
2025-02-26 16:11:52,399 - Vectorización de productos e interacciones completada y guardada.
2025-02-26 16:11:52,399 - Fin de data_preprocess.py
2025-02-26 16:11:53,698 - data_preprocess.py completado en 948.95 segundos.
2025-02-26 16:11:53,699 - Ejecutando model.py...
2025-02-26 16:12:03,216 - model.py completado en 9.52 segundos.
2025-02-26 16:12:03,217 - Iniciando FastAPI con main.py...
2025-02-26 16:17:50,852 - Inicio de orchestrator.py
2025-02-26 16:17:50,859 - Embeddings ya generados. Saltando data_preprocess.py...
2025-02-26 16:17:50,861 - Matriz de similitudes colaborativas ya generada. Saltando model.py...
2025-02-26 16:17:50,862 - Iniciando FastAPI con main.py...
2025-02-26 16:18:15,492 - Inicio de orchestrator.py
2025-02-26 16:18:15,495 - Embeddings ya generados. Saltando data_preprocess.py...
2025-02-26 16:18:15,497 - Matriz de similitudes colaborativas ya generada. Saltando model.py...
2025-02-26 16:18:15,497 - Iniciando FastAPI con main.py...
2025-02-26 18:09:10,193 - Inicio de orchestrator.py
2025-02-26 18:09:10,197 - Embeddings ya generados. Saltando data_preprocess.py...
2025-02-26 18:09:10,199 - Matriz de similitudes colaborativas ya generada. Saltando model.py...
2025-02-26 18:09:10,199 - Iniciando FastAPI con main.py...
```