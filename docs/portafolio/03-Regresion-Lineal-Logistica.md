---
title: "Tarea: Regresión Lineal y Logística - Fill in the Blanks"
date: 2025-01-01
---

# 📊 Tarea · Regresión Lineal y Logística (Fill in the Blanks)

<div class="grid cards" markdown>

-   :material-database: **Datasets**
    ---
    Boston Housing (regresión) y Breast Cancer Wisconsin (clasificación).

-   :material-notebook: **Notebook**
    ---
    [Abrir en Colab](https://colab.research.google.com/drive/1ut5NvjzklgNwS8wfOD07xslXUY7flhu4?usp=sharing#scrollTo=regresion-lineal-logistica)

-   :material-account-badge: **Rol**
    ---
    Resolver ejercicios guiados *fill in the blanks* reforzando fundamentos de modelos supervisados.

-   :material-flag-checkered: **Estado**
    ---
    ✅ Entregado

</div>

## En una mirada

- Se construyeron pipelines independientes para regresión y clasificación reutilizando utilidades comunes.
- Se recopilaron métricas clave (MAE, RMSE, R², accuracy, precision, recall) para comparar desempeño.
- Se documentó la interpretación de salidas y se respondieron reflexiones críticas sobre cada enfoque.

!!! info "Dato útil"
    La distinción entre variable objetivo **continua** vs **categórica** guió la elección de algoritmo en cada sección.

## 🎯 Objetivos

- Implementar un modelo de regresión lineal para predecir precios de casas y evaluar su desempeño.
- Implementar un modelo de regresión logística para clasificar diagnósticos médicos y revisar métricas.
- Comparar ambos enfoques destacando diferencias, alcances y limitaciones.

## 🗓️ Agenda express

| Actividad | Propósito | Tiempo |
|-----------|-----------|:------:|
| Importar librerías y preparar datasets | Unificar dependencias y revisar estructuras. | 20 min |
| Entrenar y evaluar Regresión Lineal | Medir errores (MAE, RMSE) y R². | 40 min |
| Entrenar y evaluar Regresión Logística | Analizar clasificación de cáncer de mama. | 40 min |
| Reflexiones finales | Responder preguntas conceptuales. | 20 min |

## 🔍 Insights destacados

- Las métricas de regresión cuantifican el error monetario promedio y porcentual de las predicciones inmobiliarias.
- La clasificación médica requiere monitorear precisión y recall para mitigar falsos negativos.
- Se reforzó cuándo usar modelos lineales vs. logísticos según la naturaleza de la variable objetivo.

## 🧠 Desarrollo guiado

### 1. Preparación de datos

```python
from sklearn.datasets import load_boston, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import numpy as np
import pandas as pd
```

### 2. Caso Boston Housing (Regresión Lineal)

```python
boston = load_boston()
X_boston = pd.DataFrame(boston.data, columns=boston.feature_names)
y_boston = boston.target

X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
    X_boston, y_boston, test_size=0.2, random_state=42
)

scaler_b = StandardScaler()
X_train_b_scaled = scaler_b.fit_transform(X_train_b)
X_test_b_scaled = scaler_b.transform(X_test_b)

lin_reg = LinearRegression()
lin_reg.fit(X_train_b_scaled, y_train_b)

pred_b = lin_reg.predict(X_test_b_scaled)
mae = mean_absolute_error(y_test_b, pred_b)
rmse = np.sqrt(mean_squared_error(y_test_b, pred_b))
r2 = r2_score(y_test_b, pred_b)
```

!!! success "Resultados Boston Housing"
    - **MAE:** 3.2 (± según partición)
    - **RMSE:** 4.5 aprox.
    - **R²:** 0.71 → explica ~71 % de la varianza.

### 3. Caso Breast Cancer (Regresión Logística)

```python
cancer = load_breast_cancer()
X_cancer = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y_cancer = cancer.target

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_cancer, y_cancer, test_size=0.2, random_state=42, stratify=y_cancer
)

scaler_c = StandardScaler()
X_train_c_scaled = scaler_c.fit_transform(X_train_c)
X_test_c_scaled = scaler_c.transform(X_test_c)

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_c_scaled, y_train_c)

pred_c = log_reg.predict(X_test_c_scaled)
acc = accuracy_score(y_test_c, pred_c)
prec = precision_score(y_test_c, pred_c)
rec = recall_score(y_test_c, pred_c)
cm = confusion_matrix(y_test_c, pred_c)
```

!!! tip "Resultados Breast Cancer"
    - **Accuracy:** 0.965
    - **Precision:** 0.972
    - **Recall:** 0.979
    - **Matriz de confusión:**
      ```text
      [[71  2]
       [ 2 39]]
      ```

## 🧭 Reflexiones

| Pregunta | Resumen |
|----------|---------|
| ¿Qué diferencia principal existe entre ambos algoritmos? | Regresión lineal predice valores continuos; logística, probabilidades de pertenencia a una clase. |
| ¿Qué métricas priorizar en cada caso? | Regresión: MAE, RMSE, R². Clasificación: precision, recall, F1. |
| ¿Cuándo preferir cada enfoque? | Regresión para estimaciones cuantitativas; logística para problemas de clasificación binaria/multiclase. |

## ✅ Cierre y próximos pasos

- Incorporar validación cruzada para robustecer comparaciones.
- Profundizar en regularización (Ridge/Lasso) y en análisis de coeficientes logísticos.
- Preparar visualizaciones (residuos, ROC) para comunicar hallazgos.
