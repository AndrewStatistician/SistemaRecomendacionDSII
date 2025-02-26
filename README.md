

# Sistema de Recomendaciones  Filtrado Colaborativo con Embeddings con FastAPI y SentenceTransformer

<br><br><br>

**Enfoque Seleccionado: Filtrado Colaborativo con Embeddings**
# 1. Sistema de Recomendaciones con Filtrado Colaborativo Basado en Embeddings

El sistema de recomendaciones utiliza un enfoque de **Filtrado Colaborativo basado en Embeddings** generado mediante `SentenceTransformer ('all-MiniLM-L6-v2')`. Se eligió este enfoque debido a las siguientes razones:

## Razones para Elegir Embeddings

- **Captura de Relaciones Complejas:**  
  Al utilizar embeddings, se logra captar relaciones complejas entre usuarios y productos que van más allá de simples co-ocurrencias, permitiendo recomendaciones más precisas y relevantes.

- **Generalización y Escalabilidad:**  
  Los embeddings ofrecen una mejor capacidad de generalización en comparación con técnicas tradicionales como SVD o Matrix Factorization, permitiendo una mejor adaptación a productos y usuarios nuevos.

- **Uso Eficiente de Datos de Interacciones:**  
  Se utilizan datos de interacciones para generar embeddings ponderados por el rating, capturando no solo la acción del usuario, sino también su intensidad o nivel de satisfacción.

## Razón para No Utilizar el Enfoque Híbrido

El enfoque híbrido requería combinar similitudes de contenido y colaborativas, lo cual resultó en problemas de dimensionalidad y consumo de memoria. Además, la estructura de los datos disponibles no justificaba la complejidad adicional.





## Resumen de Datos y Naturaleza de Variables
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



## Variables Utilizadas en el Modelo
La siguiente tabla detalla las **variables utilizadas en el modelo de recomendaciones**, especificando el **DataFrame** del cual provienen, las **variables seleccionadas** y la **justificación** de su inclusión. Estas variables fueron elegidas estratégicamente para **capturar tanto información descriptiva de los productos** como **el comportamiento e interacción de los usuarios**, permitiendo una generación de embeddings más **precisa y relevante**.


| DataFrame         | Variables Utilizadas                               | Justificación                                                     |
|------------------|-----------------------------------------------------|-------------------------------------------------------------------|
| `products.csv`   | `name`, `category`, `descripcion`, `palabras_clave` | Capturan información descriptiva y categórica del producto, ayudando a generar embeddings significativos. |
| `interactions.csv` | `user_id`, `product_id`, `tipo_interaccion`, `rating`, `comentario` | Se utilizan para capturar el comportamiento del usuario y ponderar los embeddings según el rating, reflejando la intensidad de la interacción. |



<br><br><br>

# **2. Arquitectura del Sistema y Tecnologías Elegidas**
El sistema de recomendaciones desarrollado utiliza una arquitectura modular y eficiente, diseñada para maximizar el rendimiento y escalar fácilmente en entornos de producción. La estructura del proyecto y las tecnologías elegidas han sido seleccionadas estratégicamente para optimizar el flujo de datos, procesamiento de texto y cálculo de similitudes, garantizando una experiencia de usuario fluida y personalizada.

## Estructura del Proyecto
La organización del proyecto sigue una arquitectura modular, dividiendo las responsabilidades en archivos específicos para mantener un código limpio, mantenible y escalable. La estructura es la siguiente:

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
## Flujo de Trabajo y Funcionalidades 🔄
Cada componente cumple un rol específico en el flujo de trabajo de recomendaciones:

- **`data_preprocess.py`:**  
  Realiza el preprocesamiento de datos, incluye **limpieza**, **normalización** y **generación de embeddings** utilizando `SentenceTransformer ('all-MiniLM-L6-v2')` para vectorizar texto de productos y comentarios de usuarios.

- **`model.py`:**  
  Implementa el **Filtrado Colaborativo** calculando **Cosine Similarity** entre embeddings, generando la **matriz de similitudes colaborativas**.

- **`main.py`:**  
  Exposición de la **API REST** con **FastAPI**, permitiendo consultar recomendaciones en **tiempo real**.

- **`orchestrator.py`:**  
  Coordina la ejecución de todos los módulos, asegurando un **flujo de datos eficiente** y **automatizando** el proceso de generación de embeddings y cálculo de similitudes.


## Tecnologías Utilizadas en el Sistema de Recomendaciones
El sistema utiliza un conjunto de tecnologías modernas diseñadas para optimizar el rendimiento, escalabilidad y precisión de las recomendaciones:


| Tecnología                  | Función Principal                                  | Características Clave                                     |
|-----------------------------|-----------------------------------------------------|-----------------------------------------------------------|
| **FastAPI**                 | Exponer la API REST de recomendaciones               | - Alto rendimiento y baja latencia <br> - Documentación interactiva (Swagger y Redoc) <br> - Integración sencilla con Docker |
| **SentenceTransformer ('all-MiniLM-L6-v2')** | Generación de embeddings de alta precisión     | - Modelo liviano y rápido <br> - Alta precisión en similitudes semánticas |
| **scikit-learn y numpy**    | Cálculos de Cosine Similarity y procesamiento de datos | - Operaciones matemáticas optimizadas <br> - Herramientas robustas para análisis de datos |
| **Docker**                  | Contenerización y despliegue en la nube              | - Portabilidad y escalabilidad <br> - Consistencia en desarrollo y producción |

<br><br><br>

# 3. Calidad del Código: Claridad, Modularidad y Buenas Prácticas

## Organización Modular y Estructura del Proyecto

El proyecto está organizado en cuatro módulos principales, cada uno con responsabilidades bien definidas:

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

# 4. Decisiones Tomadas y Pasos de Instalación/Ejecución

## Decisiones Clave en la Arquitectura del Sistema

- **Uso de Embeddings con SentenceTransformer:**  
  Se decidió utilizar `SentenceTransformer ('all-MiniLM-L6-v2')` para aprovechar su capacidad de generar representaciones densas y semánticamente ricas tanto para productos como para interacciones.

- **Filtrado Colaborativo Basado en Embeddings:**  
  Se optó por este enfoque debido a la calidad de las recomendaciones y la escalabilidad, al capturar relaciones implícitas entre usuarios y productos a partir de sus interacciones.

- **Batch Processing en Similitudes:**  
  Se utilizó un procesamiento en lotes (`batch_cosine_similarity`) para optimizar el cálculo de similitudes colaborativas y gestionar eficientemente la memoria.

## Pasos de Instalación y Ejecución

### 1. Clonar el Repositorio y Navegar al Directorio del Proyecto:

```bash
git clone https://github.com/usuario/recomendador-embeddings.git
cd recomendador-embeddings
```

### 2. Construir la Imagen de Docker:

Se utiliza un Dockerfile que define el entorno de FastAPI y las dependencias necesarias.

```bash
docker build -t recomendador-fastapi .
```
### 3. Ejecutar el Contenedor de Docker:
```
docker run -p 8080:8080 -v $(pwd)/data:/app/data recomendador-fastapi
```

### 4. Probar la API de Recomendaciones:
Se puede realizar una solicitud GET a la API expuesta:

```
curl -X GET "http://localhost:8080/recommendations?user_id=123&top_n=5"
```
o tambien  verificar en el Navegador:  http://localhost:8000/docs

Ejemplo de Respuesta:
La respuesta esperada será un JSON con las recomendaciones personalizadas:

```
{
  "user_id": 123,
  "recommendations": [
    {"product_id": "A1", "name": "Clases de Yoga Online", "category": "Salud"},
    {"product_id": "B2", "name": "Reloj Inteligente para Fitness", "category": "Deportes"}
  ]
}
```

<br><br><br>

# 10. Próximos Pasos y Mejoras Potenciales

## 1. Integración de Datos Sociodemográficos:
- **Expandir el modelo** para incluir datos de `users.csv` en futuras versiones.

## 2. Optimizaciones de Rendimiento:
- Explorar **técnicas de dimensionalidad reducida** (como PCA o TSNE) para optimizar el almacenamiento y el tiempo de cálculo de similitudes.

## 3. Expansión del Sistema:
- Añadir **recomendaciones personalizadas** basadas en tendencias o intereses emergentes utilizando **modelos secuenciales** o **redes neuronales recurrentes (RNNs)**.


# Dependencias y Flujo de Ejecución de los Módulos



<br><br><br>
<br><br><br>
<br><br><br>
<br><br><br>
<br><br><br>
<br><br><br>
<br><br><br>


---

## **2. Explicación del Dataset y Procesamiento de Datos** 

### 2.1. Datasets Utilizados



---


## **3. Tecnologías Elegidas y Arquitectura del Sistema** 🛠️

### 1. Tecnologías Utilizadas

### FastAPI
- **Función**: Exponer la API REST de recomendaciones.
- **Características**:
  - Alto rendimiento y baja latencia.
  - Documentación interactiva con Swagger y Redoc.
  - Integración sencilla con Docker para despliegue en la nube.

### SentenceTransformer ('all-MiniLM-L6-v2')
- **Función**: Generación de embeddings de alta precisión para la vectorización de texto.
- **Características**:
  - Modelo liviano y rápido.
  - Alta precisión en cálculos de similitud semántica.

### scikit-learn y numpy
- **Función**: Cálculos de Cosine Similarity para medir similitudes entre embeddings.
- **Características**:
  - Operaciones matemáticas optimizadas.
  - Herramientas robustas para el procesamiento de datos.

### Docker
- **Función**: Contenerización y despliegue en cualquier entorno.
- **Características**:
  - Portabilidad y escalabilidad.
  - Consistencia en el entorno de desarrollo y producción.

### Azure (Opcional)
- **Función**: Despliegue escalable en la nube.
- **Características**:
  - Administración de recursos en la nube.
  - Escalabilidad automática y alta disponibilidad.

---

## 4. Arquitectura del Sistema

La arquitectura modular se organiza de la siguiente manera:



### Descripción de Componentes

## app/
Contiene todos los scripts de la aplicación:

- **main.py**: Define la API con FastAPI, gestionando rutas y endpoints para obtener recomendaciones.
- **orchestrator.py**: Orquesta el flujo de datos, llamando a los módulos de preprocesamiento y modelo para generar predicciones.
- **data_preprocess.py**: Se encarga del preprocesamiento de datos, como la limpieza, concatenación de texto y generación de embeddings.
- **model.py**: Calcula las similitudes utilizando Cosine Similarity y combina los resultados en un modelo híbrido.

## data/
Contiene los datasets necesarios para el funcionamiento del modelo:

- **users.csv**: Información de usuarios.
- **products.csv**: Información de productos.
- **interactions.csv**: Historial de interacciones.

## requirements.txt
Dependencias de Python necesarias para ejecutar el proyecto.

## Dockerfile
Configuración para construir y ejecutar el contenedor de Docker, asegurando la portabilidad y consistencia del entorno de desarrollo y producción.

---

## **4. Cómo Ejecutar la API para Probar las Recomendaciones** 🛠️

### **1. Construir la Imagen de Docker**

Ejecuta el siguiente comando para construir la imagen de Docker:

### **Construcción y Ejecución con Docker:**
1. **Construir la Imagen:**
     ```
     docker build -t  recomendador-fastapi .
     ```
2. **Ejecutar el Contenedor:**
     ```
     docker run -p 8080:8080 -v $(pwd)/data:/app/data recomendador-fastapi

     ```

3. **Verificar en el Navegador:**
     ```
     http://localhost:8000/docs
     ```    



### **2. Ejecutar Localmente:**

0. **Pendiente Clonar Repositorio**

1. **Instala las dependencias:**
     ```
     pip3 install -r requirements.txt
     ```
2. **Ejecuta el Orquestador:**
     ```
     python app/orchestrator.py
     ```

3. **Abre en el navegador:**
     ```
     http://localhost:8000/docs
     ```     

# Ejemplos de Consultas a la API

### Obtener Recomendaciones para un Usuario:
```bash
curl -X GET "http://localhost:8080/recommendations?user_id=123&top_n=5"

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



```

### Servidores

- Curl

```
curl -X 'GET' \
  'http://vmdsonora:8080/recommendations?user_id=3&top_n=5' \
  -H 'accept: application/json'
```

- Request URL
```
http://vmdsonora:8080/recommendations?user_id=3&top_n=5
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
---
# Debilidades

## 1. Archivos csv

## 2. Salidas en formato .npy


# Rendimiento del modelo

## Logs



# Mejoras

## 1. Batch
```
Embeddings ya generados. Saltando data_preprocess.py...
Matriz de similitudes colaborativas ya generada. Saltando model.py...
Iniciando FastAPI con main.py...
```
El codigo como ya se corrio la noche anterior, el modelo no se calcula de nuevo si no
que da la recomendacion.
Mejora: Si tenemos algun usuario nuevo o producto o interaccion, el modelo se actualice
con un flujo de ETL que genere una actualizacion de variable PROCESAR y con esta variable
el modelo se actualice solo para los usuarios/productos que tengan esta variable en TRUE.

## 2. Algoritmos
2.1 Modelo con otros algoritmos de Sentence embedings


## 3. Tiempos


## 4. Nube

## 5. Datos en SQL csv y Bases de datos Vectoriales (Embeddings)
