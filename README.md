# Proyecto_Final_DSIII

Descripción del Proyecto 👗

Este proyecto de Análisis de Sentimiento se enfoca en las reseñas de clientes de una empresa de moda para predecir si un producto será recomendado o no. El objetivo principal es construir y optimizar un modelo de Machine Learning que transforme el texto de las reseñas en información accionable, permitiendo a la empresa comprender la satisfacción del cliente y tomar decisiones informadas sobre sus productos.
Metodología 🛠️

El proyecto sigue un enfoque estructurado de Procesamiento de Lenguaje Natural (NLP) y modelado predictivo:

1. Análisis Exploratorio de Datos (EDA): Se analizó la longitud de las reseñas, la frecuencia de palabras clave y las partes del discurso para obtener una comprensión profunda del lenguaje utilizado por los clientes.

2. Preprocesamiento de Texto: Se limpiaron, tokenizaron y lematizaron las reseñas para prepararlas para el modelado.

3. Vectorización: Se compararon dos técnicas, Bag-of-Words (BoW) y TF-IDF, para convertir el texto en un formato numérico.

4. Modelado y Optimización: Se entrenaron modelos de Regresión Logística y Naive Bayes. El modelo de Regresión Logística con Bag-of-Words se seleccionó como el de mejor rendimiento y se optimizó utilizando GridSearchCV para mejorar su precisión y su capacidad para identificar reseñas negativas.

Tecnologías Usadas 💻
Python
Pandas: Manejo y análisis de datos.
NLTK: Preprocesamiento de texto.
Scikit-learn: Modelado de Machine Learning.
Matplotlib/Seaborn: Visualización de datos.
WordCloud: Visualización de nubes de palabras.
