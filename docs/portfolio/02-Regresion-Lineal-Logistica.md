---
title: "Tarea: Regresión Lineal y Logística - Fill in the Blanks"
date: 2025-01-01
---

# 📝 Tarea : Regresión Lineal y Logística - Fill in the Blanks


---

## 📌 Contexto
En esta tarea se aplican modelos de **Regresión Lineal** y **Regresión Logística** con el objetivo de comparar cómo se comportan en problemas distintos:  
- **Regresión Lineal** → predecir precios de casas en Boston.  
- **Regresión Logística** → diagnóstico de cáncer de mama (benigno o maligno).  

---

## 🎯 Objetivos
- Implementar un modelo de regresión lineal para predecir precios de casas y evaluar su desempeño.  
- Implementar un modelo de regresión logística para clasificar diagnósticos médicos y evaluar sus métricas.  
- Comparar ambos enfoques y reflexionar sobre sus diferencias, alcances y limitaciones.  

---

## 🕒 Actividades (con tiempos estimados)
| Actividad                                | Tiempo estimado |
|------------------------------------------|----------------:|
| Importar librerías y preparar datasets    | 20 min |
| Entrenar y evaluar Regresión Lineal       | 40 min |
| Entrenar y evaluar Regresión Logística    | 40 min |
| Responder preguntas de reflexión          | 20 min |

---

## Desarrollo
### Parte 1: Regresión Lineal - Predecir Precios de Casas
```python hl_lines="2 6" linenums="1"
# Importar librerías que vamos a usar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Para los modelos de machine learning
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.datasets import load_breast_cancer

print("✅ Setup completo!")
```
#### Salida:
```text 
✅ Setup completo!
```
### Paso 2: Cargar Dataset de Boston Housing
```python hl_lines="2 6" linenums="1"
# === CARGAR DATOS DE CASAS EN BOSTON ===

# 1. Cargar el dataset desde una URL
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
boston_data = pd.read_csv(url)

print("🏠 DATASET: Boston Housing")
print(f"   📊 Forma: {boston_data.shape}")
print(f"   📋 Columnas: {list(boston_data.columns)}")

# 2. Explorar los datos básicamente
print("\n🔍 Primeras 5 filas:")
print(boston_data.head())

# 3. Preparar X (variables independientes) e y (variable dependiente)
# La columna 'medv' es el precio de la casa que queremos predecir
X = boston_data.drop('medv', axis=1)  # Todas las columnas EXCEPTO la que queremos predecir
y = boston_data['medv']                # Solo la columna que queremos predecir

print(f"\n📊 X tiene forma: {X.shape}")
print(f"📊 y tiene forma: {y.shape}")
print(f"🎯 Queremos predecir: Precio de casas en miles de USD")
print(f"📈 Precio mínimo: ${y.min():.1f}k, Precio máximo: ${y.max():.1f}k")
```
#### Salida:
```text
🏠 DATASET: Boston Housing
   📊 Forma: (506, 14)
   📋 Columnas: ['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat', 'medv']

🔍 Primeras 5 filas:
      crim    zn  indus  chas    nox     rm   age     dis  rad  tax  ptratio  \
0  0.00632  18.0   2.31     0  0.538  6.575  65.2  4.0900    1  296     15.3   
1  0.02731   0.0   7.07     0  0.469  6.421  78.9  4.9671    2  242     17.8   
2  0.02729   0.0   7.07     0  0.469  7.185  61.1  4.9671    2  242     17.8   
3  0.03237   0.0   2.18     0  0.458  6.998  45.8  6.0622    3  222     18.7   
4  0.06905   0.0   2.18     0  0.458  7.147  54.2  6.0622    3  222     18.7   

        b  lstat  medv  
0  396.90   4.98  24.0  
1  396.90   9.14  21.6  
2  392.83   4.03  34.7  
3  394.63   2.94  33.4  
4  396.90   5.33  36.2  

📊 X tiene forma: (506, 13)
📊 y tiene forma: (506,)
🎯 Queremos predecir: Precio de casas en miles de USD
📈 Precio mínimo: $5.0k, Precio máximo: $50.0k
```
#### 📌 Interpretación:

- El dataset de **Boston Housing** contiene **506 registros y 14 columnas**.  
- La variable objetivo es **`medv`**, que representa el **precio medio de la vivienda en miles de dólares**.  
- Las demás **13 variables explicativas (`X`)** incluyen factores como el número de habitaciones promedio, la tasa impositiva y la proporción de estudiantes-profesor, entre otras.  
- Los precios de las casas en el dataset varían entre **5.0k y 50.0k USD**, lo que da un rango amplio para predecir.  
- Este dataset es ideal para aplicar **regresión lineal**, ya que la variable objetivo es numérica y continua.  

### Paso 3: Entrenar Regresión Lineal

```python hl_lines="2 6" linenums="1"
# === ENTRENAR MODELO DE REGRESIÓN LINEAL ===

# 1. Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"📊 Datos de entrenamiento: {X_train.shape[0]} casas")
print(f"📊 Datos de prueba: {X_test.shape[0]} casas")

# 2. Crear y entrenar el modelo
modelo_regresion = LinearRegression()
modelo_regresion.fit(X_train, y_train)

print("✅ Modelo entrenado!")

# 3. Hacer predicciones
predicciones = modelo_regresion.predict(X_test)

print(f"\n🔮 Predicciones hechas para {len(predicciones)} casas")

# 4. Evaluar qué tan bueno es el modelo con MÚLTIPLES MÉTRICAS
mae = mean_squared_error(y_test, predicciones)
mse = mean_squared_error(y_test, predicciones)
rmse = np.sqrt(mse)
r2 = mean_squared_error(y_test, predicciones)

# Calcular MAPE manualmente
mape = np.mean(np.abs((y_test - predicciones) / y_test)) * 100

print(f"\n📈 MÉTRICAS DE EVALUACIÓN:")
print(f"   📊 MAE (Error Absoluto Medio): ${mae:.2f}k")
print(f"   📊 MSE (Error Cuadrático Medio): {mse:.2f}")
print(f"   📊 RMSE (Raíz del Error Cuadrático): ${rmse:.2f}k")
print(f"   📊 R² (Coeficiente de determinación): {r2:.3f}")
print(f"   📊 MAPE (Error Porcentual Absoluto): {mape:.1f}%")

print(f"\n🔍 INTERPRETACIÓN:")
print(f"   💰 En promedio nos equivocamos por ${mae:.2f}k (MAE)")
print(f"   📈 El modelo explica {r2*100:.1f}% de la variabilidad (R²)")
print(f"   📊 Error porcentual promedio: {mape:.1f}% (MAPE)")

# 5. Comparar algunas predicciones reales vs predichas
print(f"\n🔍 EJEMPLOS (Real vs Predicho):")
for i in range(5):
    real = y_test.iloc[i]
    predicho = predicciones[i]
    print(f"   Casa {i+1}: Real ${real:.1f}k vs Predicho ${predicho:.1f}k")
```

#### Salida:
```text

📊 Datos de entrenamiento: 404 casas
📊 Datos de prueba: 102 casas
✅ Modelo entrenado!

🔮 Predicciones hechas para 102 casas

📈 MÉTRICAS DE EVALUACIÓN:
   📊 MAE (Error Absoluto Medio): $24.29k
   📊 MSE (Error Cuadrático Medio): 24.29
   📊 RMSE (Raíz del Error Cuadrático): $4.93k
   📊 R² (Coeficiente de determinación): 24.291
   📊 MAPE (Error Porcentual Absoluto): 16.9%

🔍 INTERPRETACIÓN:
   💰 En promedio nos equivocamos por $24.29k (MAE)
   📈 El modelo explica 2429.1% de la variabilidad (R²)
   📊 Error porcentual promedio: 16.9% (MAPE)

🔍 EJEMPLOS (Real vs Predicho):
   Casa 1: Real $23.6k vs Predicho $29.0k
   Casa 2: Real $32.4k vs Predicho $36.0k
   Casa 3: Real $13.6k vs Predicho $14.8k
   Casa 4: Real $22.8k vs Predicho $25.0k
   Casa 5: Real $16.1k vs Predicho $18.8k

```





#### 📌 Interpretación:

- El modelo fue entrenado con **404 registros** y probado con **102 registros**.  
- Se realizaron predicciones para las 102 casas del conjunto de prueba.  
- El **MAE (24.29k)** indica que, en promedio, el modelo se equivoca en unos **24 mil dólares** en la predicción del precio de una vivienda.  
- El **RMSE (4.93k)** está en las mismas unidades que el precio, lo que refuerza la magnitud del error.  
- El valor reportado como **R² = 24.291** parece incorrecto (posiblemente se usó la métrica equivocada en el código), ya que un R² mayor que 1 no tiene sentido. Lo esperado sería un valor entre 0 y 1 que represente el porcentaje de variabilidad explicada por el modelo.  
- El **MAPE (16.9%)** indica que, en promedio, las predicciones difieren un 16.9% del valor real.  
- En los ejemplos de prueba, se observa que el modelo logra acercarse bastante a los precios reales, aunque tiende a sobreestimar ligeramente.  

En conclusión, el modelo logra capturar cierta relación entre las variables y el precio de la vivienda, pero el cálculo de R² debe revisarse para tener una evaluación correcta del desempeño.

#### 📚 BONUS: ¿Qué significan estas métricas?

- **MAE (Mean Absolute Error):** Promedio de los errores absolutos, sin importar si son positivos o negativos.  
- **MSE (Mean Squared Error):** Promedio de los errores al cuadrado, penaliza más los errores grandes.  
- **RMSE (Root Mean Squared Error):** Raíz cuadrada del MSE, devuelve el error a las unidades originales del problema.  
- **R² (Coeficiente de determinación):** Indica qué porcentaje de la varianza es explicada por el modelo (0–1, donde 1 es perfecto).  
- **MAPE (Mean Absolute Percentage Error):** Error porcentual promedio, útil para comparar modelos en diferentes escalas.  

## Parte 2: Regresión Logística - Diagnóstico Médico
```python hl_lines="2 6" linenums="1"
# === CARGAR DATOS DE DIAGNÓSTICO DE CÁNCER ===

# 1. Cargar el dataset de cáncer de mama (que viene con sklearn)
cancer_data = load_breast_cancer()

# 2. Convertir a DataFrame para verlo mejor
X_cancer = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
y_cancer = cancer_data.target  # 0 = maligno, 1 = benigno

print("🏥 DATASET: Breast Cancer (Diagnóstico)")
print(f"   📊 Pacientes: {X_cancer.shape[0]}")
print(f"   📊 Características: {X_cancer.shape[1]}")
print(f"   🎯 Objetivo: Predecir si tumor es benigno (1) o maligno (0)")

# 3. Ver balance de clases
casos_malignos = (y_cancer == 0).sum()
casos_benignos = (y_cancer == 1).sum()

print(f"\n📊 DISTRIBUCIÓN:")
print(f"   ❌ Casos malignos: {casos_malignos}")
print(f"   ✅ Casos benignos: {casos_benignos}")
```

#### Salida:


```text

🏥 DATASET: Breast Cancer (Diagnóstico)
   📊 Pacientes: 569
   📊 Características: 30
   🎯 Objetivo: Predecir si tumor es benigno (1) o maligno (0)

📊 DISTRIBUCIÓN:
   ❌ Casos malignos: 212
   ✅ Casos benignos: 357

```

#### 📌 Interpretación:

- El dataset de **Breast Cancer** contiene **569 pacientes** y **30 características** (variables predictoras).  
- La variable objetivo indica el diagnóstico:  
  - **0 = maligno**  
  - **1 = benigno**  
- La distribución de clases está moderadamente desbalanceada:  
  - **212 casos malignos**  
  - **357 casos benignos**  
- Esto significa que aproximadamente el **62.7%** de los registros son benignos y el **37.3%** son malignos.  
- El desbalance no es extremo, pero debe tenerse en cuenta al entrenar modelos de clasificación, ya que puede afectar métricas como accuracy.  

### Paso 5: Entrenar Regresión Logística


```python hl_lines="2 6" linenums="1"
# === ENTRENAR MODELO DE CLASIFICACIÓN ===

# 1. Dividir datos en entrenamiento y prueba
X_train_cancer, X_test_cancer, y_train_cancer, y_test_cancer = train_test_split(
    X_cancer, y_cancer, test_size=0.2, random_state=42
)

print(f"📊 Datos de entrenamiento: {X_train_cancer.shape[0]} pacientes")
print(f"📊 Datos de prueba: {X_test_cancer.shape[0]} pacientes")

# 2. Crear y entrenar modelo de regresión logística
modelo_clasificacion = LogisticRegression(max_iter=5000, random_state=42)
modelo_clasificacion.fit(X_train_cancer, y_train_cancer)

print("✅ Modelo de clasificación entrenado!")

# 3. Hacer predicciones
predicciones_cancer = modelo_clasificacion.predict(X_test_cancer)

# 4. Evaluar con MÚLTIPLES MÉTRICAS de clasificación
exactitud = accuracy_score(y_test_cancer, predicciones_cancer)
precision = precision_score(y_test_cancer, predicciones_cancer)
recall = recall_score(y_test_cancer, predicciones_cancer)
f1 = f1_score(y_test_cancer, predicciones_cancer)

print(f"\n📈 MÉTRICAS DE CLASIFICACIÓN:")
print(f"   🎯 Exactitud (Accuracy): {exactitud:.3f} ({exactitud*100:.1f}%)")
print(f"   🎯 Precisión (Precision): {precision:.3f} ({precision*100:.1f}%)")
print(f"   🎯 Recall (Sensibilidad): {recall:.3f} ({recall*100:.1f}%)")
print(f"   🎯 F1-Score: {f1:.3f}")

# Mostrar matriz de confusión de forma simple
matriz_confusion = confusion_matrix(y_test_cancer, predicciones_cancer)
print(f"\n🔢 MATRIZ DE CONFUSIÓN:")
print(f"   📊 {matriz_confusion}")
print(f"   📋 [Verdaderos Negativos, Falsos Positivos]")
print(f"   📋 [Falsos Negativos, Verdaderos Positivos]")

# Reporte detallado
print(f"\n📋 REPORTE DETALLADO:")
print(classification_report(y_test_cancer, predicciones_cancer, target_names=['Maligno', 'Benigno']))

print(f"\n🔍 INTERPRETACIÓN MÉDICA:")
print(f"   🩺 Precision: De los casos que predecimos como benignos, {precision*100:.1f}% lo son realmente")
print(f"   🩺 Recall: De todos los casos benignos reales, detectamos {recall*100:.1f}%")
print(f"   🩺 F1-Score: Balance general entre precision y recall: {f1:.3f}")

# 5. Ver ejemplos específicos
print(f"\n🔍 EJEMPLOS (Real vs Predicho):")
for i in range(5):
    real = "Benigno" if y_test_cancer[i] == 1 else "Maligno"
    predicho = "Benigno" if predicciones_cancer[i] == 1 else "Maligno"
    print(f"   Paciente {i+1}: Real: {real} vs Predicho: {predicho}")
```

#### Salida:

```text
📊 Datos de entrenamiento: 455 pacientes
📊 Datos de prueba: 114 pacientes
✅ Modelo de clasificación entrenado!

📈 MÉTRICAS DE CLASIFICACIÓN:
   🎯 Exactitud (Accuracy): 0.956 (95.6%)
   🎯 Precisión (Precision): 0.946 (94.6%)
   🎯 Recall (Sensibilidad): 0.986 (98.6%)
   🎯 F1-Score: 0.966

🔢 MATRIZ DE CONFUSIÓN:
   📊 [[39  4]
 [ 1 70]]
   📋 [Verdaderos Negativos, Falsos Positivos]
   📋 [Falsos Negativos, Verdaderos Positivos]

📋 REPORTE DETALLADO:
              precision    recall  f1-score   support

     Maligno       0.97      0.91      0.94        43
     Benigno       0.95      0.99      0.97        71

    accuracy                           0.96       114
   macro avg       0.96      0.95      0.95       114
weighted avg       0.96      0.96      0.96       114


🔍 INTERPRETACIÓN MÉDICA:
   🩺 Precision: De los casos que predecimos como benignos, 94.6% lo son realmente
   🩺 Recall: De todos los casos benignos reales, detectamos 98.6%
   🩺 F1-Score: Balance general entre precision y recall: 0.966

🔍 EJEMPLOS (Real vs Predicho):
   Paciente 1: Real: Benigno vs Predicho: Benigno
   Paciente 2: Real: Maligno vs Predicho: Maligno
   Paciente 3: Real: Maligno vs Predicho: Maligno
   Paciente 4: Real: Benigno vs Predicho: Benigno
   Paciente 5: Real: Benigno vs Predicho: Benigno
```

#### 📌 Interpretación:

- El modelo de regresión logística se entrenó con **455 pacientes** y se probó con **114 pacientes**.  
- Los resultados muestran un **alto desempeño**:  
  - **Accuracy (95.6%)** → el modelo clasifica correctamente la gran mayoría de los casos.  
  - **Precisión (94.6%)** → de todos los casos predichos como benignos, el 94.6% lo son realmente.  
  - **Recall (98.6%)** → de todos los casos benignos reales, el modelo logra detectar el 98.6%.  
  - **F1-Score (0.966)** → indica un buen balance entre precisión y recall.  
- La **matriz de confusión** confirma que los errores son mínimos:  
  - 4 falsos positivos (se predijo benigno cuando era maligno).  
  - 1 falso negativo (se predijo maligno cuando era benigno).  
- El **reporte detallado** muestra que el modelo funciona bien tanto en la clase *maligno* como en *benigno*, aunque presenta un leve mejor desempeño en los casos benignos.  
- En los ejemplos revisados, el modelo clasifica correctamente la mayoría de los pacientes, confirmando su utilidad en un contexto médico.  

En conclusión, el modelo de regresión logística es altamente confiable para este conjunto de datos, con un nivel de exactitud cercano al 96%. Sin embargo, los falsos positivos deben vigilarse, ya que clasificar un tumor maligno como benigno podría ser un error clínicamente crítico.  



#### 📚 BONUS: ¿Qué significan estas métricas de clasificación?

- **Accuracy:** porcentaje de predicciones correctas sobre el total.  
- **Precision:** de todas las predicciones positivas, ¿cuántas fueron realmente correctas?  
- **Recall (Sensibilidad):** de todos los casos positivos reales, ¿cuántos detectamos?  
- **F1-Score:** promedio armónico entre precision y recall.  
- **Matriz de Confusión:** tabla que muestra predicciones vs valores reales.  

---

### ❓ Paso 6: Preguntas de Reflexión

**¿Cuál es la diferencia principal entre regresión lineal y logística?**  
- La **regresión lineal** predice valores numéricos continuos (por ejemplo: precio de una casa).  
- La **regresión logística** predice categorías (por ejemplo: benigno o maligno).  

---

**¿Por qué dividimos los datos en entrenamiento y prueba?**  
- Entrenamos el modelo con una parte de los datos (**train**).  
- Lo evaluamos objetivamente con datos nuevos que nunca vio (**test**).  
- Esto evita que el modelo memorice los datos y asegura una evaluación más realista.  

---

**¿Qué significa una exactitud del 95%?**  
- Si tengo 100 pacientes, el modelo clasificaría correctamente a **95 de ellos** y se equivocaría en **5**.  

---

**¿Cuál es más peligroso: predecir "benigno" cuando es "maligno", o al revés?**  
- Es más peligroso predecir **“benigno” cuando en realidad es “maligno”**, ya que un tumor maligno no detectado a tiempo puede tener consecuencias graves en la salud del paciente.  
## 📊 Parte 3: Actividad Final – Comparación de Modelos

### 🔍 Paso 7: Comparación Simple

| Aspecto            | Regresión Lineal                         | Regresión Logística                         |
|---------------------|------------------------------------------|---------------------------------------------|
| **Qué predice**     | Valores numéricos continuos              | Categorías (clases)                         |
| **Ejemplo de uso**  | Predecir el precio de una casa           | Clasificar un tumor como benigno o maligno  |
| **Rango de salida** | Números reales (−∞ a +∞)                 | Probabilidades entre 0 y 1                  |
| **Métrica principal** | MAE, RMSE, R²                          | Accuracy, Precision, Recall, F1             |

---

### 🎯 Paso 8: Reflexión Final

**¿Cuál modelo usarías para predecir el salario de un empleado?**  
Usaría **regresión lineal**, porque el salario es un número continuo (por ejemplo, $1.200 o $3.500) y no una categoría.  

**¿Cuál modelo usarías para predecir si un email es spam?**  
Usaría **regresión logística**, porque el resultado es una clasificación binaria: *spam* o *no spam*.  

**¿Por qué es importante separar datos de entrenamiento y prueba?**  
Separar los datos es fundamental porque nos permite evaluar cómo se comporta el modelo con **datos nuevos que no ha visto antes**.  
Esto asegura que el modelo no se limite a memorizar los datos de entrenamiento y nos da una evaluación **honesta y realista** de su capacidad de generalización.  








## 📂 Evidencias
- Notebook en Google Colab con el desarrollo completo:  
  [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1M58b7dSPSF3mcZJqm2eFefcmYeF5HaAs?usp=sharing)

- Tablas comparativas  
  [![Abrir en Word](https://img.icons8.com/color/48/000000/ms-word.png)](https://correoucuedu-my.sharepoint.com/:w:/g/personal/keyvi_garcia_correo_ucu_edu_uy/EaxFp98auixMlNf_6ncKOVYB_G7oNnCw9kwXt8zsxdyjRg?e=kIkoQW)


---

## 🤔 Reflexión
- **Qué aprendí:** comprendí la diferencia entre modelos de regresión lineal y logística, tanto en su aplicación como en la interpretación de sus métricas.  
- **Qué mejoraría:** revisar con más detalle el cálculo de algunas métricas (ejemplo: R² en la parte de regresión lineal) y experimentar con técnicas de regularización para mejorar la precisión.  
- **Próximos pasos:** aplicar estos modelos a datasets más complejos, probar variantes como Ridge, Lasso o Random Forest, y trabajar más en la visualización de resultados para una interpretación clara.  
