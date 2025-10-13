---
title: "📚 Portafolio de Prácticas y Proyectos"
date: 2025-10-14
---

# 🚀 Portafolio de Prácticas y Proyectos  
> _"Cada práctica es un desafío para conectar teoría, código y creatividad."_  

Bienvenido/a a mi portafolio académico.  
Aquí registro todas las prácticas y proyectos del curso, documentando **objetivos, metodología, resultados, evidencias y reflexiones**.  

---

# 🧭 UT1 — Machine Learning Clásico  
**Del análisis exploratorio a la validación de modelos**

---

<div class="grid cards" markdown>

### 🧩 Exploración del Titanic: patrones de supervivencia con EDA  
:material-chart-scatter-plot: **Práctica 1 — EDA del Titanic (Colab)**  
Exploración de datos, calidad, valores faltantes y patrones de supervivencia.  
[:octicons-arrow-right-24: Ver práctica](01-primera-entrada.md){ .md-button .md-button--primary }  

**Trabajo extra:**  
:material-plus: *EDA comparativa: Titanic vs. naufragios modernos (simulados)*  
[:octicons-file-24: Ver artículo](01x-titanic-extra.md){ .md-button }

---

### ⚙️ Cómo mejorar un modelo predictivo con Feature Engineering  
:material-cog: **Práctica 2 — Feature Engineering + Modelo Base**  
Imputación, variables dummies, *feature crosses* y baseline con regresión logística.  
[:octicons-arrow-right-24: Ver práctica](02-Feature-Engineering.md){ .md-button .md-button--primary }  

**Trabajo extra:**  
:material-plus: *Feature Store mini: reproducibilidad y comparación de modelos*  
[:octicons-file-24: Ver artículo](02x-feature-extra.md){ .md-button }

---

### 📈 Regresión lineal y logística: del modelo a la interpretación  
:material-chart-line: **Práctica 3 — Regresión Lineal y Logística**  
Comparación entre modelos, métricas y optimización del umbral de decisión.  
[:octicons-arrow-right-24: Ver práctica](02-Regresion-Lineal-Logistica.md){ .md-button .md-button--primary }  

**Trabajo extra:**  
:material-plus: *Regresión robusta + Clasificación con umbral óptimo (ROC/PR)*  
[:octicons-file-24: Ver artículo](03x-regresion-extra.md){ .md-button }

---

### 🧠 Validación y Selección de Modelos: buscando el mejor candidato  
:material-check-decagram: **Práctica 4 — Validación y Selección de Modelos**  
*Pipelines*, *KFold/StratifiedKFold*, *GridSearchCV* y *RandomizedSearchCV*.  
[:octicons-arrow-right-24: Ver práctica](03-Validacion-Seleccion-deModelos.md){ .md-button .md-button--primary }  

**Trabajo extra:**  
:material-plus: *Torneo extendido: XGBoost/LightGBM + curvas de aprendizaje*  
[:octicons-file-24: Ver artículo](04x-model-selection-extra.md){ .md-button }

---

### 🧮 Segmentación inteligente: Clustering y PCA en acción  
:material-account-multiple: **Práctica 5 — Clustering y PCA (Mall Customers)**  
K-Means, métricas *Silhouette* y PCA para visualización de clusters.  
[:octicons-arrow-right-24: Ver práctica](04-Clustering-PCA-Mall.md){ .md-button .md-button--primary }  

**Trabajo extra:**  
:material-plus: *Clustering avanzado: GMM/DBSCAN + validación de estabilidad*  
[:octicons-file-24: Ver artículo](05x-clustering-extra.md){ .md-button }

</div>

---

# 🤖 UT2 — Deep Learning Foundations  
**De perceptrones simples a redes profundas optimizadas**

---

<div class="grid cards" markdown>

### 🔌 Del perceptrón a las redes neuronales modernas  
:material-robot-excited-outline: **Práctica 7 — De Perceptrón a Redes Neuronales**  
Implementación de perceptrón, MLP y exploración de activaciones y pesos.  
[:octicons-arrow-right-24: Ver práctica](07-Perceptron-MLP.md){ .md-button .md-button--primary }  

**Trabajo extra:**  
:material-plus: *Activaciones y optimizadores: análisis de impacto en la convergencia*  
[:octicons-file-24: Ver artículo](07x-deep-learning-extra.md){ .md-button }

---

### 🧪 Experimentación y control del entrenamiento  
:material-chart-bar-stacked: **Práctica 8 — Experimentación con MLPs**  
Entrenamiento, curvas de pérdida, *callbacks* y evaluación final de modelos.  
[:octicons-arrow-right-24: Ver práctica](08-Experimentacions.md){ .md-button .md-button--primary }  

**Trabajos extra:**  
- :material-plus: *Regularización avanzada: L1/L2, Dropout y Batch Normalization*  
  [:octicons-file-24: Ver artículo](08x-experimentacion-extra.md){ .md-button }  

- :material-plus: *Experimentación de arquitecturas MLP (profundidad y ancho)*  
  [:octicons-file-24: Ver artículo](08b-deep-learning-extra.md){ .md-button }  

- :material-plus: *Optimizadores e hiperparámetros: comparación de Adam, SGD y AdamW*  
  [:octicons-file-24: Ver artículo](08c-Optimizadores-MLP.md){ .md-button }  

- :material-plus: *Callbacks en MLPs: EarlyStopping, LR Scheduler y ModelCheckpoint*  
  [:octicons-file-24: Ver artículo](08d-Callbacks-MLP.md){ .md-button }

</div>

---
## 🧪 Ideas específicas para los **Trabajos extra**

??? note "Práctica 1 — Titanic (EDA)"
    - **Dataset alternativo:** [Airbnb listings](https://insideairbnb.com/get-the-data) o accidentes de aviación (Kaggle).  
    - **Objetivo:** replicar *pipeline* de EDA (faltantes, outliers, correlaciones) y contar **insights accionables**.  
    - **Plus:** *data story* con 3–5 visualizaciones narrativas.

??? note "Práctica 2 — Feature Engineering"
    - **Nuevas features:** `FamilySize`, `IsChild`, `CabinDeck`, *target encoding* controlado (sin leakage).  
    - **Comparación:** LR vs. Árboles (DT/Random Forest) con el *mismo* set de features.  
    - **Plus:** guarda las features con `joblib` para reproducibilidad (*mini feature store*).

??? note "Práctica 3 — Regresión y Logística"
    - **Regresión robusta:** Huber/Quantile para mitigar outliers (BostonHousing u otro dataset).  
    - **Clasificación:** optimiza **umbral** con ROC/PR según costo de errores (FN > FP, etc.).  
    - **Plus:** informe de **errores críticos** con casos ejemplo.

??? note "Práctica 4 — Selección de Modelos"
    - **Torneo:** agrega XGBoost/LightGBM y compara estabilidad (σ baja en CV).  
    - **Curvas de aprendizaje:** *under/overfitting* vs. tamaño de datos.  
    - **Plus:** matriz de **riesgos** (complejidad ↔ interpretabilidad).

??? note "Práctica 5 — Clustering y PCA"
    - **Algoritmos:** GMM con BIC/AIC, DBSCAN con *grid* de eps/min_samples.  
    - **Estabilidad:** *bootstrap clustering* o ARI/NMI.  
    - **Plus:** perfil de **segmentos** con acciones de marketing por cluster.

??? note "Práctica 7 — De Perceptrón a Redes Neuronales"
    - **Dataset:** Utiliza el dataset MNIST o Fashion MNIST.
    - **Objetivo:** Construir y entrenar un Perceptrón simple para clasificación binaria.
    - **Plus:** Experimenta con las funciones de activación (`relu`, `sigmoid`, `tanh`) y observa cómo afecta la convergencia y el rendimiento en Keras/TensorFlow.

??? note "Práctica 8 — Experimentación"
    - **Regularización:** Aplica capas de `Dropout` y `BatchNormalization` y compáralas con un modelo base sin regularización.
    - **Callbacks:** Implementa `EarlyStopping` y `ModelCheckpoint` y documenta su impacto en el tiempo de entrenamiento y la calidad del modelo.
    - **Plus:** Visualiza el historial de pérdida y precisión (*accuracy*) para identificar sobreajuste (*overfitting*).

---

## 🔄 Flujo de documentación

```mermaid
graph LR
    A[📌 Preparación] --> B[⚙️ Ejecución]
    B --> C[📈 Evaluación]
    C --> D[📷 Evidencias]
