# 🎵 Spotify Reviews Sentiment Analysis using NLP and Deep Learning

## 📌 Descripción del Proyecto

Este proyecto desarrolla un pipeline completo de Procesamiento de Lenguaje Natural (NLP) y Deep Learning para clasificar automáticamente reseñas de usuarios de Spotify en sentimientos positivos o negativos.

El trabajo integra todas las etapas de un proyecto real de clasificación de texto:

- Análisis Exploratorio de Datos (EDA)
- Limpieza y preprocesamiento textual
- Tokenización y lematización con NLTK y spaCy
- Representación numérica mediante TF-IDF
- Implementación de una Red Neuronal en PyTorch
- Entrenamiento con técnicas de regularización
- Evaluación e interpretación de resultados

---

## 🎯 Objetivos

Construir un modelo de clasificación de texto capaz de identificar el sentimiento de una reseña de Spotify utilizando técnicas de NLP y Deep Learning.

Objetivos específicos:

- Analizar la distribución de sentimientos en las reseñas.
- Aplicar técnicas de limpieza y normalización textual.
- Comparar la lematización utilizando NLTK y spaCy.
- Transformar texto en variables numéricas mediante TF-IDF.
- Entrenar una red neuronal para clasificación binaria.
- Evaluar el desempeño del modelo mediante métricas de clasificación.

---

## 📂 Dataset

### Spotify App Reviews 2022

Fuente:

https://www.kaggle.com/datasets/mfaaris/spotify-app-reviews-2022

El dataset contiene reseñas reales de usuarios de Spotify obtenidas desde Google Play Store.

Variables originales:

| Variable | Descripción |
|-----------|-------------|
| Time_submitted | Fecha de la reseña |
| Review | Texto de la reseña |
| Rating | Calificación otorgada (1 a 5 estrellas) |
| Total_thumbsup | Cantidad de votos positivos |
| Reply | Respuesta oficial de Spotify |

Cantidad total de registros:

**Más de 60.000 reseñas**

---

## 🧹 Preprocesamiento

Se implementó un pipeline completo de limpieza textual:

### 1. Conversión a minúsculas

Normalización básica para evitar duplicidades léxicas.

Ejemplo:

```text
Spotify Is Great
```

↓

```text
spotify is great
```

### 2. Limpieza con Expresiones Regulares (Regex)

Eliminación de:

- URLs
- Caracteres especiales
- Números
- Signos de puntuación

### 3. Tokenización

Se realizó una comparación entre:

- NLTK (`word_tokenize`)
- spaCy (`en_core_web_sm`)

### 4. Lematización

Se compararon dos enfoques:

- WordNetLemmatizer (NLTK)
- Lemmatizer de spaCy

La lematización con spaCy mostró resultados más consistentes para verbos conjugados y contracciones.

### 5. Eliminación de Stopwords

Se eliminaron palabras de baja capacidad discriminativa como:

```text
the
and
is
are
to
of
```

---

## 📊 Análisis Exploratorio de Datos (EDA)

Se realizaron los siguientes análisis:

- Distribución de ratings.
- Distribución de sentimientos.
- Longitud de las reseñas.
- Palabras más frecuentes.
- Visualizaciones descriptivas.

---

## 🔢 Representación Numérica

### TF-IDF (Term Frequency - Inverse Document Frequency)

Se utilizó:

```python
TfidfVectorizer(max_features=5000)
```

### ¿Por qué TF-IDF?

TF-IDF permite ponderar la importancia de cada término considerando:

- Frecuencia dentro del documento.
- Frecuencia global en el corpus.

De esta manera se reducen los pesos de palabras muy comunes y se destacan términos más relevantes para la clasificación.

---

## 🤖 Modelo de Deep Learning

Se implementó una red neuronal utilizando PyTorch.

### Arquitectura

```text
Input (5000 características TF-IDF)
        ↓
Linear (5000 → 512)
        ↓
Batch Normalization
        ↓
Dropout (0.5)
        ↓
ReLU
        ↓
Linear (512 → 256)
        ↓
Batch Normalization
        ↓
Dropout (0.5)
        ↓
ReLU
        ↓
Linear (256 → 2)
```

### Componentes utilizados

- PyTorch
- Batch Normalization
- Dropout (p = 0.5)
- Adam Optimizer
- CrossEntropyLoss
- Early Stopping

---

## ⚙️ Entrenamiento

Configuración:

```python
learning_rate = 0.001
batch_size = 128
epochs = 50
patience = 5
```

Se implementó Early Stopping para evitar sobreajuste cuando la pérdida de validación no mejora durante cinco épocas consecutivas.

---

## 📈 Resultados

### Accuracy

```text
87.99%
```

### Classification Report

| Clase | Precision | Recall | F1-Score |
|---------|---------:|---------:|---------:|
| Negative | 0.85 | 0.89 | 0.87 |
| Positive | 0.91 | 0.87 | 0.89 |

### Matriz de Confusión

```text
[[4406  548]
 [ 766 5222]]
```

---

## 🔍 Análisis de Resultados

El modelo obtuvo un desempeño satisfactorio para una tarea de clasificación binaria de texto.

Se observó un fenómeno de overfitting durante el entrenamiento:

- La pérdida de entrenamiento disminuyó continuamente.
- La pérdida de validación comenzó a aumentar después de las primeras épocas.

Para mitigar este comportamiento se utilizó Early Stopping, conservando automáticamente el modelo con mejor desempeño sobre el conjunto de validación.

---

## ⚠️ Limitaciones

- Algunas reseñas contienen opiniones mixtas difíciles de clasificar.
- El modelo utiliza únicamente TF-IDF, sin información contextual profunda.
- La red neuronal mostró señales de sobreajuste.

---

## 🚀 Mejoras Futuras

Posibles extensiones del proyecto:

- Word Embeddings (Word2Vec, GloVe).
- Redes LSTM o GRU.
- Modelos Transformer (BERT).
- Optimización de hiperparámetros.
- Técnicas adicionales de regularización.

---

## 🛠️ Tecnologías Utilizadas

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- NLTK
- spaCy
- Scikit-Learn
- PyTorch

---

Proyecto desarrollado como Trabajo Final de NLP y Deep Learning para Ciencia de Datos.
