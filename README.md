# 👗 E-commerce Review Sentiment Analysis: Recommendation Prediction

## 👒Project Overview

This data science project focuses on Binary Classification using a dataset containing 23,486 customer reviews from a women's fashion e-commerce company. The goal is to build a Machine Learning model that predicts whether a customer will recommend a product (Recommended IND) based solely on the review text.
The work involved applying robust Natural Language Processing (NLP) techniques and developing and optimizing a Logistic Regression model to achieve high predictive accuracy while managing the challenge of class imbalance.

## 👒Business and Analytical Context

### - Business Context
To understand the root causes of product recommendation or non-recommendation, thereby improving product quality, design, and marketing strategy. The goal is to convert free-text reviews into actionable business intelligence.
### - Analytical Problem
Binary Classification. Predict the Recommended IND variable (1 = Recommended, 0 = Not Recommended).
### - Target Audience
Product Managers, Marketing Teams, and Data Analysts.

## 👒Question and Hypothesis

- Question: Can we predict with high accuracy whether a customer will recommend a product based solely on their review text?
- Hypothesis (H1) Accepted: It is possible to build an ML model with an overall accuracy greater than 85% and a recall for the minority class (Not Recommended, class 0) greater than 50%. The final model achieved an Overall Accuracy of 89% and a Class 0 Recall of 57%.

## 👒Tech Stack

- Language: Python
- Libraries: Pandas, NLTK, Scikit-learn, Matplotlib, Seaborn
- ML/NLP Techniques: Lexical Processing (Tokenization, Lemmatization), POS Tagging, N-gram Analysis (up to 4-grams), Bag of Words (BoW) Models, Logistic Regression, Naive Bayes, GridSearchCV.

## 👒Natural Language Processing (NLP)

A rigorous preprocessing pipeline was applied to prepare the text data:

1. Cleaning: Lowercasing, URL, and punctuation removal (retaining only letters).
2. Tokenization & Lemmatization: Words were reduced to their base form (lemmas).
3. Lexical Analysis/POS Tagging: Confirmed that Adjectives (JJ) and Nouns (NN) are the most influential parts of speech for sentiment.
4. N-grams Analysis: Revealed key phrases like ('run', 'large') (dissatisfaction) and ('fit', 'perfectly') (satisfaction).

## 👒Modeling Results

The dataset is imbalanced (82% Recommended vs. 18% Not Recommended), leading to a focus on maximizing Recall for Class 0 (Not Recommended).

### 👚Base Model Comparison (BoW vs. TF-IDF)

The Logistic Regression model using Bag of Words (BoW) showed a superior capability in identifying the minority class (negative reviews) compared to the TF-IDF version.

- Logistic Regression with BoW: Achieved a Class 0 Recall of 0.51 and an Overall Accuracy of 0.8842.
- Logistic Regression with TF-IDF: Resulted in a lower Class 0 Recall (0.39) and Overall Accuracy (0.8708).

Conclusion: The BoW model was selected for the optimization phase due to its better performance on the minority class.

### 👚Performance of Final Optimized Model

- GridSearchCV was used on the Logistic Regression model with BoW encoding to tune the C (regularization) and ngram_range hyperparameters.
- Best Hyperparameters: C: 100, ngram_range: (1, 3)

The optimized model achieved an Overall Accuracy of 89% with the following key metrics:

#### Class 0 (Not Recommended):
- Recall: 0.57
- Precision: 0.75
#### Class 1 (Recommended):
- Recall: 0.96
- Precision: 0.91

## 👒Final Conclusions

1. Hypothesis Acceptance (H1): The review text is a significant predictor. The optimized Logistic Regression model is robust and meets the project objectives for both accuracy (89% > 85%) and Class 0 recall (57% > 50%).
2. Key Driver Identification: Exploratory analysis and modeling confirmed critical words for each sentiment:
- Dissatisfaction Indicators (Class 0): fit, small, fabric, disappointed, poor.
- Satisfaction Indicators (Class 1): love, perfect, great, beautiful, exactly.
3. Business Value: These findings provide direct, actionable insights, allowing Product and Design teams to prioritize critical issues such as sizing/fit and material quality, which are the dominant drivers of customer dissatisfaction.

# 👗 Análisis de Sentimiento de Reseñas de E-commerce: Predicción de Recomendación

## 👒Resumen del Proyecto

Este proyecto aborda un problema de Clasificación Binaria utilizando un conjunto de datos de 23,486 reseñas de clientes de una empresa de comercio electrónico de moda femenina. El objetivo es construir un modelo de Machine Learning que prediga si un cliente recomendará o no un producto (Recommended IND) basándose únicamente en el texto libre de su reseña (Review Text).
Se aplicaron técnicas robustas de Procesamiento de Lenguaje Natural (NLP) y se desarrolló y optimizó un modelo de Regresión Logística para lograr una alta precisión predictiva, superando el reto del desequilibrio de clases.

## 👒Contexto Comercial y Analítico

- Contexto Comercial: Comprender las razones profundas detrás de la recomendación o no recomendación de productos para mejorar la calidad, el diseño y la estrategia de marketing. El objetivo es convertir las reseñas de texto libre en información accionable.
- Problema Analítico: Clasificación Binaria. Predecir Recommended IND (1 = Recomendado, 0 = No Recomendado).
- Audiencia: Gerentes de Producto, Equipos de Marketing y Analistas de Datos.

## 👒Pregunta e Hipótesis

- Pregunta: ¿Podemos predecir con alta precisión si un cliente recomendará un producto basándonos únicamente en el texto de su reseña?
- Hipótesis (H1) Aceptada: Es posible construir un modelo de ML con una precisión general superior al 85% y un recall para la clase minoritaria (No Recomendado, clase 0) superior al 50%. El modelo final logró una Precisión Global del 89% y un Recall de la Clase 0 del 57%.

## 👒Stack Técnico

- Lenguaje: Python
- Librerías: Pandas, NLTK, Scikit-learn, Matplotlib, Seaborn
- Técnicas ML/NLP: Procesamiento Léxico (Tokenización, Lematización), POS Tagging, Análisis de N-gramas (hasta 4-gramas), Modelos Bag of Words (BoW), Regresión Logística, Naive Bayes.

## 👒Procesamiento de Lenguaje Natural (NLP)
Se aplicó un pipeline de preprocesamiento riguroso para preparar el texto:

1. Limpieza: Conversión a minúsculas, eliminación de URL y puntuación (conservando solo letras).
2. Tokenización y Lematización: Se descompusieron las reseñas en tokens y se redujeron las palabras a su forma base (lemas).
3. Análisis Léxico/POS Tagging: Confirmó que los Adjetivos (JJ) y Sustantivos (NN) son las partes del discurso más influyentes para determinar el sentimiento.
4. Análisis de N-gramas: Reveló frases clave de insatisfacción como ('run', 'large') y de satisfacción como ('fit', 'perfectly').

## 👒Resultados del Modelado

El conjunto de datos es desequilibrado (82% Recomendado, 18% No Recomendado), por lo que la optimización se centró en mejorar el Recall para la Clase 0 (No Recomendado).

### 👚Comparativa de Modelos Base (BoW vs. TF-IDF)

La regresión logística con Bag of Words (BoW) fue superior a la versión con TF-IDF, especialmente para identificar la clase minoritaria (No Recomendado).

- Regresión Logística con BoW: Logró un Recall para la Clase 0 de 0.51 y una Precisión Global de 0.8842.
- Regresión Logística con TF-IDF: Obtuvo un Recall de la Clase 0 de 0.39 y una Precisión Global de 0.8708.
Conclusión: El modelo BoW mostró una capacidad superior para capturar la clase minoritaria (reseñas negativas) y fue seleccionado para la fase de optimización.

### 👚Rendimiento del Modelo Final Optimizado

Se utilizó GridSearchCV sobre la Regresión Logística con codificación BoW, optimizando los hiperparámetros C (regularización) y ngram_range.
#### Mejores Hiperparámetros: C: 100, ngram_range: (1, 3)

El modelo optimizado alcanzó una Precisión Global del 89% y métricas específicas clave:
#### Clase 0 (No Recomendado):
- Recall: 0.57
- Precision: 0.75
#### Clase 1 (Recomendado):
- Recall: 0.96
- Precision: 0.91

## 👒Conclusiones

1. Aceptación de H1: Se demostró que el texto de las reseñas es un predictor significativo. El modelo de Regresión Logística optimizado es robusto y cumple con los objetivos de precisión (89% > 85%) y recall de la clase 0 (57% > 50%).
2. Identificación de Drivers: El análisis exploratorio identificó palabras clave críticas para cada sentimiento:
- Indicadores de Insatisfacción (Clase 0): fit, small, fabric, disappointed, poor.
- Indicadores de Satisfacción (Clase 1): love, perfect, great, beautiful, exactly.
3. Valor Comercial: Estos hallazgos permiten a los equipos de Producto y Diseño priorizar mejoras en aspectos como el tallaje y la calidad de los materiales, que son los principales drivers de la insatisfacción de los clientes.
