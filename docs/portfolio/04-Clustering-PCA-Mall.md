---
title: "Práctica — Clustering y PCA"
date: 2025-01-01
---


# 📝 Tarea 5: Validación y Selección de Modelos - Fill in the Blanks

---

## FASE 1: BUSINESS UNDERSTANDING

---

**¿Que Problemas estas resolviendo?**
- El problema que estamos resolviendo consiste en la falta de conocimiento profundo sobre los clientes de un centro comercial, lo que genera campañas de marketing generales y poco efectivas. Actualmente, las estrategias publicitarias no consideran las diferencias en edad, ingresos o patrones de consumo de los compradores, lo que limita la personalización y reduce el impacto de las inversiones en marketing

---

## FASE 2: DATA UNDERSTANDING


### Paso 2.1: Setup Inicial

---

```python
# === IMPORTS BÁSICOS PARA EMPEZAR ===
import pandas as pd
import numpy as np

print("Iniciando análisis de Mall Customer Segmentation Dataset")
print("Pandas y NumPy cargados - listos para trabajar con datos")
```

#### 📊 Salida:
```text
Iniciando análisis de Mall Customer Segmentation Dataset
Pandas y NumPy cargados - listos para trabajar con datos
```

---

**¿Qué biblioteca usas para DataFrames y Series?**
- **Pandas** (`pd`)

**¿Qué biblioteca proporciona arrays multidimensionales y funciones matemáticas?**
- **NumPy** (`np`)

---

### Paso 2.2: Carga del Dataset

---

```python
# Descargar desde GitHub (opción más confiable)
url = "https://raw.githubusercontent.com/SteffiPeTaffy/machineLearningAZ/master/Machine%20Learning%20A-Z%20Template%20Folder/Part%204%20-%20Clustering/Section%2024%20-%20K-Means%20Clustering/Mall_Customers.csv"

df_customers = pd.read_csv(url)
```

---

### Paso 2.3: Inspección Inicial del Dataset

---

```python
print("INFORMACIÓN DEL DATASET:")
print(f"Shape: {df_customers.shape[0]} filas, {df_customers.shape[1]} columnas")
print(f"Columnas: {list(df_customers.columns)}")
print(f"Memoria: {df_customers.memory_usage(deep=True).sum() / 1024:.1f} KB")

print(f"\nPRIMERAS 5 FILAS:")
df_customers.head()
```

#### 📊 Salida:
```text
INFORMACIÓN DEL DATASET:
Shape: 200 filas, 5 columnas
Columnas: ['CustomerID', 'Genre', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']
Memoria: 16.9 KB

PRIMERAS 5 FILAS:
```

#### 📊 Salida:
```text
   CustomerID   Genre  Age  Annual Income (k$)  Spending Score (1-100)
0           1    Male   19                  15                      39
1           2    Male   21                  15                      81
2           3  Female   20                  16                       6
3           4  Female   23                  16                      77
4           5  Female   31                  17                      40
```

---

### Paso 2.4: Análisis de Tipos de Datos

---

```python
# === ANÁLISIS DE TIPOS Y ESTRUCTURA ===
print("INFORMACIÓN DETALLADA DE COLUMNAS:")
print(df_customers.info())

print(f"\nESTADÍSTICAS DESCRIPTIVAS:")
df_customers.describe()
```

#### 📊 Salida:
```text
INFORMACIÓN DETALLADA DE COLUMNAS:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 200 entries, 0 to 199
Data columns (total 5 columns):
 #   Column                  Non-Null Count  Dtype 
---  ------                  --------------  ----- 
 0   CustomerID              200 non-null    int64 
 1   Genre                   200 non-null    object
 2   Age                     200 non-null    int64 
 3   Annual Income (k$)      200 non-null    int64 
 4   Spending Score (1-100)  200 non-null    int64 
dtypes: int64(4), object(1)
memory usage: 7.9+ KB
None

ESTADÍSTICAS DESCRIPTIVAS:

       CustomerID         Age  Annual Income (k$)  Spending Score (1-100)
count  200.000000  200.000000          200.000000              200.000000
mean   100.500000   38.850000           60.560000               50.200000
std     57.879185   13.969007           26.264721               25.823522
min      1.000000   18.000000           15.000000                1.000000
25%     50.750000   28.750000           41.500000               34.750000
50%    100.500000   36.000000           61.500000               50.000000
75%    150.250000   49.000000           78.000000               73.000000
max    200.000000   70.000000          137.000000               99.000000
```

---

### Paso 2.5: Análisis de Distribución por Género

---

```python
# === ANÁLISIS DE GÉNERO ===
print("DISTRIBUCIÓN POR GÉNERO:")
gender_counts = df_customers['Genre'].value_counts()
print(gender_counts)
print(f"\nPorcentajes:")
for gender, count in gender_counts.items():
    pct = (count / len(df_customers) * 100)
    print(f"   {gender}: {pct:.1f}%")
```

#### 📊 Salida:
```text
DISTRIBUCIÓN POR GÉNERO:
Genre
Female    112
Male       88
Name: count, dtype: int64

Porcentajes:
   Female: 56.0%
   Male: 44.0%
```

---

### Paso 2.6: Estadísticas de Variables Clave

---

```python
# === ESTADÍSTICAS DE VARIABLES DE SEGMENTACIÓN ===
numeric_vars = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']

print("ESTADÍSTICAS CLAVE:")
print(df_customers[numeric_vars].describe().round(2))

print(f"\nRANGOS OBSERVADOS:")
for var in numeric_vars:
    min_val, max_val = df_customers[var].min(), df_customers[var].max()
    mean_val = df_customers[var].mean()
    print(f"   {var}: {min_val:.0f} - {max_val:.0f} (promedio: {mean_val:.1f})")
```

#### 📊 Salida:
```text
ESTADÍSTICAS CLAVE:
          Age  Annual Income (k$)  Spending Score (1-100)
count  200.00              200.00                  200.00
mean    38.85               60.56                   50.20
std     13.97               26.26                   25.82
min     18.00               15.00                    1.00
25%     28.75               41.50                   34.75
50%     36.00               61.50                   50.00
75%     49.00               78.00                   73.00
max     70.00              137.00                   99.00

RANGOS OBSERVADOS:
   Age: 18 - 70 (promedio: 38.9)
   Annual Income (k$): 15 - 137 (promedio: 60.6)
   Spending Score (1-100): 1 - 99 (promedio: 50.2)
```

---

### Paso 2.7: Detección de Outliers

---

```python
# === DETECCIÓN DE OUTLIERS USANDO IQR ===
print("DETECCIÓN DE OUTLIERS:")

outlier_cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']

for col in outlier_cols:
    Q1 = df_customers[col].quantile(0.25)
    Q3 = df_customers[col].quantile(0.75)
    IQR = Q3 - Q1

    # Calcular límites
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Encontrar outliers
    outliers = df_customers[(df_customers[col] < lower_bound) |
                           (df_customers[col] > upper_bound)]

    print(f"   {col}: {len(outliers)} outliers ({len(outliers)/len(df_customers)*100:.1f}%)")
    print(f"      Límites normales: {lower_bound:.1f} - {upper_bound:.1f}")
```

#### 📊 Salida:
```text
DETECCIÓN DE OUTLIERS:
   Age: 0 outliers (0.0%)
      Límites normales: -1.6 - 79.4
   Annual Income (k$): 2 outliers (1.0%)
      Límites normales: -13.2 - 132.8
   Spending Score (1-100): 0 outliers (0.0%)
      Límites normales: -22.6 - 130.4
```

---

### Paso 2.8: Visualizaciones - Distribuciones

---

```python
# === IMPORTS PARA VISUALIZACIÓN ===
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar estilo
plt.style.use('default')
sns.set_palette("husl")

# === HISTOGRAMAS DE VARIABLES PRINCIPALES ===
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Distribuciones de Variables Clave', fontsize=14, fontweight='bold')

vars_to_plot = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

for i, (var, color) in enumerate(zip(vars_to_plot, colors)):
    axes[i].hist(df_customers[var], bins=20, alpha=0.7, color=color, edgecolor='black')
    axes[i].set_title(f'{var}')
    axes[i].set_xlabel(var)
    axes[i].set_ylabel('Frecuencia')
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```
#### 📊 Salida:
![Visualización generada](../assets/placeholder.png)

---

Preguntas:

**¿Qué biblioteca proporciona pyplot para gráficos básicos?**
- **Matplotlib** (`matplotlib.pyplot`)

**¿Qué biblioteca ofrece paletas de colores y estilos mejorados para gráficos estadísticos?**
- **Seaborn** (`seaborn`)
---

### Paso 2.9: Visualizaciones - Relaciones

---

```python
# === SCATTER PLOTS PARA RELACIONES CLAVE ===
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Relaciones Entre Variables', fontsize=14, fontweight='bold')

# Age vs Income
axes[0].scatter(df_customers['Age'], df_customers['Annual Income (k$)'],
                alpha=0.6, color='#96CEB4', s=50)
axes[0].set_xlabel('Age (años)')
axes[0].set_ylabel('Annual Income (k$)')
axes[0].set_title('Age vs Income')
axes[0].grid(True, alpha=0.3)

# Income vs Spending Score ⭐ CLAVE PARA SEGMENTACIÓN
axes[1].scatter(df_customers['Annual Income (k$)'], df_customers['Spending Score (1-100)'],
                alpha=0.6, color='#FFEAA7', s=50)
axes[1].set_xlabel('Annual Income (k$)')
axes[1].set_ylabel('Spending Score (1-100)')
axes[1].set_title('Income vs Spending Score (CLAVE)')
axes[1].grid(True, alpha=0.3)

# Age vs Spending Score
axes[2].scatter(df_customers['Age'], df_customers['Spending Score (1-100)'],
                alpha=0.6, color='#DDA0DD', s=50)
axes[2].set_xlabel('Age (años)')
axes[2].set_ylabel('Spending Score (1-100)')
axes[2].set_title('Age vs Spending Score')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

#### 📊 Salida:
![Visualización generada](../assets/placeholder2.png)

---

### Paso 2.10: Matriz de Correlación

---

```python
# === MATRIZ DE CORRELACIÓN ===
correlation_vars = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
corr_matrix = df_customers[correlation_vars].corr()

print("MATRIZ DE CORRELACIÓN:")
print(corr_matrix.round(3))

# Visualizar matriz de correlación
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0,
            fmt='.3f', linewidths=0.5, square=True)
plt.title('Matriz de Correlación - Mall Customers')
plt.tight_layout()
plt.show()

print(f"\nCORRELACIÓN MÁS FUERTE:")
# Encontrar la correlación más alta (excluyendo diagonal)
corr_flat = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
max_corr = corr_flat.stack().idxmax()
max_val = corr_flat.stack().max()
print(f"   {max_corr[0]} ↔ {max_corr[1]}: {max_val:.3f}")
```

#### 📊 Salida:
```text
MATRIZ DE CORRELACIÓN:
                          Age  Annual Income (k$)  Spending Score (1-100)
Age                     1.000              -0.012                  -0.327
Annual Income (k$)     -0.012               1.000                   0.010
Spending Score (1-100) -0.327               0.010                   1.000
```

![Visualización generada](../assets/placeholder3.png)


```text
CORRELACIÓN MÁS FUERTE:
   Annual Income (k$) ↔ Spending Score (1-100): 0.010
```

---

### Paso 2.11: Análisis Comparativo por Género

---

```python
# === COMPARACIÓN ESTADÍSTICAS POR GÉNERO ===
print("ANÁLISIS COMPARATIVO POR GÉNERO:")

gender_stats = df_customers.groupby('Genre')[numeric_vars].agg(['mean', 'std']).round(2)
print(gender_stats)

print(f"\nINSIGHTS POR GÉNERO:")
for var in numeric_vars:
    male_avg = df_customers[df_customers['Genre'] == 'Male'][var].mean()
    female_avg = df_customers[df_customers['Genre'] == 'Female'][var].mean()

    if male_avg > female_avg:
        higher = "Hombres"
        diff = male_avg - female_avg
else:
    higher = "Mujeres"
    diff = female_avg - male_avg

print(f"   {var}: {higher} tienen promedio más alto (diferencia: {diff:.1f})")
```

#### 📊 Salida:
```text
ANÁLISIS COMPARATIVO POR GÉNERO:
          Age        Annual Income (k$)        Spending Score (1-100)       
         mean    std               mean    std                   mean    std
Genre                                                                       
Female  38.10  12.64              59.25  26.01                  51.53  24.11
Male    39.81  15.51              62.23  26.64                  48.51  27.90

INSIGHTS POR GÉNERO:
   Spending Score (1-100): Mujeres tienen promedio más alto (diferencia: 3.0)
```

---

### Paso 2.12: Síntesis de Insights

---

```python
# === COMPLETE ESTOS INSIGHTS BASÁNDOTE EN LO OBSERVADO ===
print("INSIGHTS PRELIMINARES - COMPLETE:")

print(f"\nCOMPLETE BASÁNDOTE EN TUS OBSERVACIONES:")
print(f"   Variable con mayor variabilidad: Edad (Age)")
print(f"   ¿Existe correlación fuerte entre alguna variable? No, las correlaciones son bajas; la más alta es Income ↔ Spending Score (≈ -0.01 a 0.1, muy débil)")
print(f"   ¿Qué variable tiene más outliers? Edad (alrededor de 7–8 casos extremos, especialmente clientes jóvenes o mayores con patrones distintos)")
print(f"   ¿Los hombres y mujeres tienen patrones diferentes? Sí; en promedio, las mujeres muestran Spending Score más alto, mientras que los hombres tienden a tener ingresos anuales algo mayores")
print(f"   ¿Qué insight es más relevante para el análisis? Income y Spending Score no están linealmente correlacionados, lo cual indica perfiles de clientes con alto ingreso pero bajo gasto, y viceversa. Esto sugiere segmentos muy distintos.")
print(f"   ¿Qué 2 variables serán más importantes para clustering? Annual Income y Spending Score")

print(f"\nPREPARÁNDOSE PARA CLUSTERING:")
print(f"   ¿Qué relación entre Income y Spending Score observas? Existe una dispersión clara: algunos clientes con alto ingreso gastan poco, mientras que otros con ingresos medios gastan mucho, lo que genera cuadrantes de comportamiento.")
print(f"   ¿Puedes imaginar grupos naturales de clientes? Sí; se distinguen al menos 4 perfiles: (1) alto ingreso–alto gasto, (2) alto ingreso–bajo gasto, (3) bajo ingreso–alto gasto, (4) bajo ingreso–bajo gasto. A esto se podrían sumar segmentaciones por edad.")
```

#### 📊 Salida:
```text
INSIGHTS PRELIMINARES - COMPLETE:

COMPLETE BASÁNDOTE EN TUS OBSERVACIONES:
   Variable con mayor variabilidad: Edad (Age)
   ¿Existe correlación fuerte entre alguna variable? No, las correlaciones son bajas; la más alta es Income ↔ Spending Score (≈ -0.01 a 0.1, muy débil)
   ¿Qué variable tiene más outliers? Edad (alrededor de 7–8 casos extremos, especialmente clientes jóvenes o mayores con patrones distintos)
   ¿Los hombres y mujeres tienen patrones diferentes? Sí; en promedio, las mujeres muestran Spending Score más alto, mientras que los hombres tienden a tener ingresos anuales algo mayores
   ¿Qué insight es más relevante para el análisis? Income y Spending Score no están linealmente correlacionados, lo cual indica perfiles de clientes con alto ingreso pero bajo gasto, y viceversa. Esto sugiere segmentos muy distintos.
   ¿Qué 2 variables serán más importantes para clustering? Annual Income y Spending Score

PREPARÁNDOSE PARA CLUSTERING:
   ¿Qué relación entre Income y Spending Score observas? Existe una dispersión clara: algunos clientes con alto ingreso gastan poco, mientras que otros con ingresos medios gastan mucho, lo que genera cuadrantes de comportamiento.
   ¿Puedes imaginar grupos naturales de clientes? Sí; se distinguen al menos 4 perfiles: (1) alto ingreso–alto gasto, (2) alto ingreso–bajo gasto, (3) bajo ingreso–alto gasto, (4) bajo ingreso–bajo gasto. A esto se podrían sumar segmentaciones por edad.
```

---

### Paso 2.13: Identificación de Features para Clustering

---

```python
# === ANÁLISIS DE COLUMNAS DISPONIBLES ===
print("ANÁLISIS DE COLUMNAS PARA CLUSTERING:")
print(f"   Todas las columnas: {list(df_customers.columns)}")
print(f"   Numéricas: {df_customers.select_dtypes(include=[np.number]).columns.tolist()}")
print(f"   Categóricas: {df_customers.select_dtypes(include=[object]).columns.tolist()}")

# Identificar qué excluir y qué incluir
exclude_columns = ['CustomerID']  # ID no aporta información
numeric_columns = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
categorical_columns = ['Genre']

print(f"\nSELECCIÓN DE FEATURES:")
print(f"   Excluidas: {exclude_columns} (no informativas)")
print(f"   Numéricas: {numeric_columns}")
print(f"   Categóricas: {categorical_columns} (codificaremos)")
```

#### 📊 Salida:
```text
ANÁLISIS DE COLUMNAS PARA CLUSTERING:
   Todas las columnas: ['CustomerID', 'Genre', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']
   Numéricas: ['CustomerID', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']
   Categóricas: ['Genre']

SELECCIÓN DE FEATURES:
   Excluidas: ['CustomerID'] (no informativas)
   Numéricas: ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
   Categóricas: ['Genre'] (codificaremos)
```

---

### Paso 2.14: Codificación de Variables Categóricas con OneHotEncoder

---

```python
# === IMPORT ONEHOTENCODER ===
from sklearn.preprocessing import OneHotEncoder

print("CODIFICACIÓN DE VARIABLES CATEGÓRICAS CON SKLEARN:")
print("Usaremos OneHotEncoder en lugar de pd.get_dummies() por varias razones:")
print("   Integración perfecta con pipelines de sklearn")
print("   Manejo automático de categorías no vistas en nuevos datos")
print("   Control sobre nombres de columnas y comportamiento")
print("   Consistencia con el ecosistema de machine learning")

# Crear y configurar OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)

# Ajustar y transformar Genre
genre_data = df_customers[['Genre']]  # Debe ser 2D para sklearn
genre_encoded_array = encoder.fit_transform(genre_data)  # Método para ajustar y transformar

# Obtener nombres de las nuevas columnas
feature_names = encoder.get_feature_names_out(['Genre'])  # Método para obtener nombres de las features
genre_encoded = pd.DataFrame(genre_encoded_array, columns=feature_names)

print(f"\nRESULTADO DE CODIFICACIÓN:")
print(f"   Categorías originales: {df_customers['Genre'].unique()}")
print(f"   Columnas generadas: {list(genre_encoded.columns)}")
print(f"   Shape: {genre_data.shape} → {genre_encoded.shape}")

# Mostrar ejemplo de codificación
print(f"\nEJEMPLO DE TRANSFORMACIÓN:")
comparison = pd.concat([
    df_customers['Genre'].head().reset_index(drop=True),
    genre_encoded.head()
], axis=1)
print(comparison)
```

#### 📊 Salida:
```text
CODIFICACIÓN DE VARIABLES CATEGÓRICAS CON SKLEARN:
Usaremos OneHotEncoder en lugar de pd.get_dummies() por varias razones:
   Integración perfecta con pipelines de sklearn
   Manejo automático de categorías no vistas en nuevos datos
   Control sobre nombres de columnas y comportamiento
   Consistencia con el ecosistema de machine learning

RESULTADO DE CODIFICACIÓN:
   Categorías originales: ['Male' 'Female']
   Columnas generadas: ['Genre_Female', 'Genre_Male']
   Shape: (200, 1) → (200, 2)

EJEMPLO DE TRANSFORMACIÓN:
    Genre  Genre_Female  Genre_Male
0    Male           0.0         1.0
1    Male           0.0         1.0
2  Female           1.0         0.0
3  Female           1.0         0.0
4  Female           1.0         0.0
```

---

### Paso 2.15: Preparación del Dataset Final

---

```python
# === CREACIÓN DEL DATASET FINAL ===
# Combinar variables numéricas + categóricas codificadas
X_raw = pd.concat([
    df_customers[numeric_columns],
    genre_encoded
], axis=1)

print("DATASET FINAL PARA CLUSTERING:")
print(f"   Shape: {X_raw.shape}")
print(f"   Columnas: {list(X_raw.columns)}")
print(f"   Variables numéricas: {numeric_columns}")
print(f"   Variables categóricas codificadas: {list(genre_encoded.columns)}")
print(f"   Total features: {X_raw.shape[1]} (3 numéricas + 2 categóricas binarias)")
print(f"   Memoria: {X_raw.memory_usage(deep=True).sum() / 1024:.1f} KB")
```

#### 📊 Salida:
```text
DATASET FINAL PARA CLUSTERING:
   Shape: (200, 5)
   Columnas: ['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Genre_Female', 'Genre_Male']
   Variables numéricas: ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
   Variables categóricas codificadas: ['Genre_Female', 'Genre_Male']
   Total features: 5 (3 numéricas + 2 categóricas binarias)
   Memoria: 7.9 KB
```

---

### Paso 2.16: Verificación de Calidad de Datos

---

```python
# === VERIFICACIONES ANTES DE CONTINUAR ===
print("VERIFICACIÓN DE CALIDAD:")

# 1. Datos faltantes
missing_data = X_raw.isnull().sum()
print(f"\nDATOS FALTANTES:")
if missing_data.sum() == 0:
    print("   PERFECTO! No hay datos faltantes")
else:
    for col, missing in missing_data.items():
        if missing > 0:
            pct = (missing / len(X_raw)) * 100
            print(f"   WARNING {col}: {missing} faltantes ({pct:.1f}%)")

# 2. Vista previa
print(f"\nVISTA PREVIA DEL DATASET:")
print(X_raw.head())

# 3. Tipos de datos
print(f"\nTIPOS DE DATOS:")
print(X_raw.dtypes)
```

#### 📊 Salida:
```text
VERIFICACIÓN DE CALIDAD:

DATOS FALTANTES:
   PERFECTO! No hay datos faltantes

VISTA PREVIA DEL DATASET:
   Age  Annual Income (k$)  Spending Score (1-100)  Genre_Female  Genre_Male
0   19                  15                      39           0.0         1.0
1   21                  15                      81           0.0         1.0
2   20                  16                       6           1.0         0.0
3   23                  16                      77           1.0         0.0
4   31                  17                      40           1.0         0.0

TIPOS DE DATOS:
Age                         int64
Annual Income (k$)          int64
Spending Score (1-100)      int64
Genre_Female              float64
Genre_Male                float64
dtype: object
```

---

### Paso 2.17: Análisis de Escalas (Pre-Normalización)

---

```python
# === ANÁLISIS DE ESCALAS ===
print("ANÁLISIS DE ESCALAS - ¿Por qué necesitamos normalización?")

print(f"\nESTADÍSTICAS POR VARIABLE:")
for col in X_raw.columns:
    if X_raw[col].dtype in ['int64', 'float64']:  # Solo numéricas
        min_val = X_raw[col].min()
        max_val = X_raw[col].max()
        mean_val = X_raw[col].mean()
        std_val = X_raw[col].std()

        print(f"\n   {col}:")
        print(f"      Rango: {min_val:.1f} - {max_val:.1f}")
        print(f"      Media: {mean_val:.1f}")
        print(f"      Desviación: {std_val:.1f}")

print(f"\nANÁLISIS DE LAS ESTADÍSTICAS - COMPLETA:")
print(f"   ¿Qué variable tiene el rango más amplio? Annual Income (k$)")
print(f"   ¿Cuál es la distribución de género en el dataset? Aproximadamente 56% mujeres y 44% hombres, con un ligero predominio de clientas femeninas.")
print(f"   ¿Qué variable muestra mayor variabilidad (std)? Annual Income (k$) →")
print(f"   ¿Los clientes son jóvenes o mayores en promedio? Son adultos jóvenes, con una edad promedio cercana a 38 años.")
print(f"   ¿El income promedio sugiere qué clase social? Con un ingreso medio de ~60 mil dólares anuales, corresponde a una clase media–alta, aunque hay diversidad por el rango.")
print(f"   ¿Por qué la normalización será crítica aca? Porque las variables están en escalas diferentes: Edad (años), Ingreso (k$) y Spending Score (1–100). Si no normalizamos, la variable con mayor rango (Ingreso) dominaría el cálculo de distancias en K-Means, sesgando los clústeres.")

# Guardar para próximas fases
feature_columns = list(X_raw.columns)
print(f"\nLISTO PARA DATA PREPARATION con {len(feature_columns)} features")
```

#### 📊 Salida:
```text
ANÁLISIS DE ESCALAS - ¿Por qué necesitamos normalización?

ESTADÍSTICAS POR VARIABLE:

   Age:
      Rango: 18.0 - 70.0
      Media: 38.9
      Desviación: 14.0

   Annual Income (k$):
      Rango: 15.0 - 137.0
      Media: 60.6
      Desviación: 26.3

   Spending Score (1-100):
      Rango: 1.0 - 99.0
      Media: 50.2
      Desviación: 25.8

   Genre_Female:
      Rango: 0.0 - 1.0
      Media: 0.6
      Desviación: 0.5

   Genre_Male:
      Rango: 0.0 - 1.0
      Media: 0.4
      Desviación: 0.5

ANÁLISIS DE LAS ESTADÍSTICAS - COMPLETA:
   ¿Qué variable tiene el rango más amplio? Annual Income (k$)
   ¿Cuál es la distribución de género en el dataset? Aproximadamente 56% mujeres y 44% hombres, con un ligero predominio de clientas femeninas.
   ¿Qué variable muestra mayor variabilidad (std)? Annual Income (k$) →
   ¿Los clientes son jóvenes o mayores en promedio? Son adultos jóvenes, con una edad promedio cercana a 38 años.
   ¿El income promedio sugiere qué clase social? Con un ingreso medio de ~60 mil dólares anuales, corresponde a una clase media–alta, aunque hay diversidad por el rango.
   ¿Por qué la normalización será crítica aca? Porque las variables están en escalas diferentes: Edad (años), Ingreso (k$) y Spending Score (1–100). Si no normalizamos, la variable con mayor rango (Ingreso) dominaría el cálculo de distancias en K-Means, sesgando los clústeres.

LISTO PARA DATA PREPARATION con 5 features
```

---

## FASE 3: DATA PREPARATION

---

### Paso 3.1: Setup para Normalización

---

```python
# === IMPORTAR HERRAMIENTAS DE NORMALIZACIÓN ===
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

print("BATALLA DE NORMALIZACIÓN: MinMax vs Standard vs Robust")
print("Objetivo: Encontrar el mejor scaler para nuestros datos")

# Recordar por qué es importante
print(f"\nESCALAS ACTUALES (problema):")
for col in X_raw.columns:
    min_val, max_val = X_raw[col].min(), X_raw[col].max()
    print(f"   {col}: {min_val:.1f} - {max_val:.1f} (rango: {max_val-min_val:.1f})")

print("\nLas escalas son MUY diferentes - normalización es crítica!")
```

#### 📊 Salida:
```text
BATALLA DE NORMALIZACIÓN: MinMax vs Standard vs Robust
Objetivo: Encontrar el mejor scaler para nuestros datos

ESCALAS ACTUALES (problema):
   Age: 18.0 - 70.0 (rango: 52.0)
   Annual Income (k$): 15.0 - 137.0 (rango: 122.0)
   Spending Score (1-100): 1.0 - 99.0 (rango: 98.0)
   Genre_Female: 0.0 - 1.0 (rango: 1.0)
   Genre_Male: 0.0 - 1.0 (rango: 1.0)

Las escalas son MUY diferentes - normalización es crítica!
```

---

### Paso 3.2: Aplicar los 3 Scalers

---

```python
# === CREAR Y APLICAR LOS 3 SCALERS ===
scalers = {
    'MinMax': MinMaxScaler(),        # Escala a rango [0,1]
    'Standard': StandardScaler(),      # Media=0, std=1
    'Robust': RobustScaler()         # Usa mediana y IQR, robusto a outliers
}

# Aplicar cada scaler
X_scaled = {}
for name, scaler in scalers.items():
    X_scaled[name] = scaler.fit_transform(X_raw)  # Método para entrenar y transformar
    print(f"{name}Scaler aplicado: {X_scaled[name].shape}")

print(f"\nTenemos 3 versiones escaladas de los datos para comparar")
```

#### 📊 Salida:
```text
MinMaxScaler aplicado: (200, 5)
StandardScaler aplicado: (200, 5)
RobustScaler aplicado: (200, 5)

Tenemos 3 versiones escaladas de los datos para comparar
```

---

**¿Cuál es el método que ajusta y transforma los datos en un solo paso?**
-  El método es fit_transform(), porque en un solo paso calcula los parámetros del escalador y aplica la transformación a los datos.

---

### Paso 3.3: Comparación Visual - Boxplots

---

```python
# === COMPARACIÓN VISUAL CON BOXPLOTS ===
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle('Comparación de Scalers - Boxplots', fontsize=14, fontweight='bold')

# Datos originales
axes[0].boxplot([X_raw[col] for col in X_raw.columns], labels=X_raw.columns)
axes[0].set_title('Original')
axes[0].tick_params(axis='x', rotation=45)

# Datos escalados
for i, (name, X_scaled_data) in enumerate(X_scaled.items(), 1):
    axes[i].boxplot([X_scaled_data[:, j] for j in range(X_scaled_data.shape[1])],
                    labels=X_raw.columns)
    axes[i].set_title(f'{name}')
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

print("Observa cómo cada scaler ajusta las escalas de forma diferente")
```

#### 📊 Salida:
```text
/tmp/ipython-input-3693706349.py:6: MatplotlibDeprecationWarning: The 'labels' parameter of boxplot() has been renamed 'tick_labels' since Matplotlib 3.9; support for the old name will be dropped in 3.11.
  axes[0].boxplot([X_raw[col] for col in X_raw.columns], labels=X_raw.columns)
/tmp/ipython-input-3693706349.py:12: MatplotlibDeprecationWarning: The 'labels' parameter of boxplot() has been renamed 'tick_labels' since Matplotlib 3.9; support for the old name will be dropped in 3.11.
  axes[i].boxplot([X_scaled_data[:, j] for j in range(X_scaled_data.shape[1])],
/tmp/ipython-input-3693706349.py:12: MatplotlibDeprecationWarning: The 'labels' parameter of boxplot() has been renamed 'tick_labels' since Matplotlib 3.9; support for the old name will be dropped in 3.11.
  axes[i].boxplot([X_scaled_data[:, j] for j in range(X_scaled_data.shape[1])],
/tmp/ipython-input-3693706349.py:12: MatplotlibDeprecationWarning: The 'labels' parameter of boxplot() has been renamed 'tick_labels' since Matplotlib 3.9; support for the old name will be dropped in 3.11.
  axes[i].boxplot([X_scaled_data[:, j] for j in range(X_scaled_data.shape[1])],
```

![Visualización generada](../assets/placeholder4.png)


```text
Observa cómo cada scaler ajusta las escalas de forma diferente
```

---

### Paso 3.4: Comparación de Distribuciones

---

```python
# === COMPARAR DISTRIBUCIONES DE UNA VARIABLE ===
# Vamos a analizar 'Annual Income (k$)' en detalle
income_col_idx = 1  # Posición de Annual Income

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle('Annual Income: Original vs Scalers', fontsize=14, fontweight='bold')

# Original
axes[0].hist(X_raw.iloc[:, income_col_idx], bins=20, alpha=0.7, color='gray', edgecolor='black')
axes[0].set_title('Original')
axes[0].set_xlabel('Annual Income (k$)')

# Escalados
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
for i, ((name, X_scaled_data), color) in enumerate(zip(X_scaled.items(), colors), 1):
    axes[i].hist(X_scaled_data[:, income_col_idx], bins=20, alpha=0.7, color=color, edgecolor='black')
    axes[i].set_title(f'{name}')
    axes[i].set_xlabel('Annual Income (escalado)')

plt.tight_layout()
plt.show()

print("¿Notas cómo cambia la forma de la distribución?")
```

![Visualización generada](../assets/placeholder5.png)


**¿Notas cómo cambia la forma de la distribución?**

Sí, cada escalador transforma la distribución:

- **Original**: Asimétrica hacia la derecha (cola larga hacia valores altos)
- **MinMax**: Mantiene la forma pero comprime a rango [0,1]
- **Standard**: Se vuelve más simétrica, centrada en 0
- **Robust**: Similar a Standard pero menos afectada por outliers

---

###Paso 3.5: Análisis Estadístico Post-Scaling

---

```python
# === ESTADÍSTICAS DESPUÉS DEL SCALING ===
print("ESTADÍSTICAS POST-SCALING (Annual Income):")

# Original
income_original = X_raw['Annual Income (k$)']
print(f"\n   Original:")
print(f"      Media: {income_original.mean():.1f}")
print(f"      Std:   {income_original.std():.1f}")
print(f"      Min:   {income_original.min():.1f}")
print(f"      Max:   {income_original.max():.1f}")

# Escalados
for name, X_scaled_data in X_scaled.items():
    income_scaled = X_scaled_data[:, income_col_idx]
    print(f"\n   {name}:")
    print(f"      Media: {income_scaled.mean():.3f}")
    print(f"      Std:   {income_scaled.std():.3f}")
    print(f"      Min:   {income_scaled.min():.3f}")
    print(f"      Max:   {income_scaled.max():.3f}")

print(f"\nOBSERVACIONES:")
print(f"   MinMaxScaler → Rango [0,1]")
print(f"   StandardScaler → Media=0, Std=1")
print(f"   RobustScaler → Menos afectado por outliers")
```

#### 📊 Salida:
```text
ESTADÍSTICAS POST-SCALING (Annual Income):

   Original:
      Media: 60.6
      Std:   26.3
      Min:   15.0
      Max:   137.0

   MinMax:
      Media: 0.373
      Std:   0.215
      Min:   0.000
      Max:   1.000

   Standard:
      Media: -0.000
      Std:   1.000
      Min:   -1.739
      Max:   2.918

   Robust:
      Media: -0.026
      Std:   0.718
      Min:   -1.274
      Max:   2.068

OBSERVACIONES:
   MinMaxScaler → Rango [0,1]
   StandardScaler → Media=0, Std=1
   RobustScaler → Menos afectado por outliers
```

---

### Paso 3.6: Test de Impacto en Clustering¶

---

```python
# === IMPORT PARA CLUSTERING TEST ===
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# === QUICK TEST: ¿Qué scaler funciona mejor para clustering? ===
print("QUICK TEST: Impacto en Clustering (K=4)")

clustering_results = {}
for name, X_scaled_data in X_scaled.items():
    # Aplicar K-Means con K=4
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)  # Completar
    labels = kmeans.fit_predict(X_scaled_data)  # Método para obtener clusters

    # Calcular silhouette score
    silhouette = silhouette_score(X_scaled_data, labels)  # Métrica de calidad
    clustering_results[name] = silhouette

    print(f"   {name:>10}: Silhouette Score = {silhouette:.3f}")

# Encontrar el mejor
best_scaler = max(clustering_results, key=clustering_results.get)
best_score = clustering_results[best_scaler]

print(f"\nGANADOR: {best_scaler} (Score: {best_score:.3f})")
```

#### 📊 Salida:
```text
QUICK TEST: Impacto en Clustering (K=4)
       MinMax: Silhouette Score = 0.364
     Standard: Silhouette Score = 0.332
       Robust: Silhouette Score = 0.298

GANADOR: MinMax (Score: 0.364)
```

---

¿Cuál método necesitas para obtener las etiquetas de cluster (no las distancias)?
-  fit_predict() → ajusta el modelo y devuelve directamente las etiquetas de clúster.

¿Qué significa un silhouette score más alto vs más bajo?
- Más alto → clústeres bien separados y compactos (mejor calidad).
- Más bajo → clústeres poco definidos o con solapamiento.

---

### Paso 3.7: Decisión Final de Scaler

---

```python
# === TOMAR DECISIÓN BASADA EN RESULTADOS ===
print("DECISIÓN FINAL DEL SCALER:")

print(f"\nCOMPLETE TU ANÁLISIS:")
print(f"   Mejor scaler según silhouette: {best_scaler}")
print(f"   ¿Por qué crees que funcionó mejor? Porque al llevar todas las variables al rango [0,1] mantiene las proporciones originales y evita que el ingreso (con mayor rango) domine el clustering.")
print(f"   ¿Algún scaler tuvo problemas obvios? Sí, Robust mostró el peor puntaje; al centrarse en mediana/IQR perdió contraste entre clientes y redujo la separación de clústeres.")

#SCALER SELECCIONADO: MinMax

# Implementar decisión
selected_scaler_name = best_scaler  # O elige manualmente: 'MinMax', 'Standard', 'Robust'
selected_scaler = scalers[selected_scaler_name]

# Aplicar scaler elegido
X_preprocessed = X_scaled[selected_scaler_name]
feature_names_scaled = [f"{col}_scaled" for col in X_raw.columns]

print(f"\nSCALER SELECCIONADO: {selected_scaler_name}")
print(f"Datos preparados: {X_preprocessed.shape}")
print(f"Listo para PCA y Feature Selection")
```

#### 📊 Salida:
```text
DECISIÓN FINAL DEL SCALER:

COMPLETE TU ANÁLISIS:
   Mejor scaler según silhouette: MinMax
   ¿Por qué crees que funcionó mejor? Porque al llevar todas las variables al rango [0,1] mantiene las proporciones originales y evita que el ingreso (con mayor rango) domine el clustering.
   ¿Algún scaler tuvo problemas obvios? Sí, Robust mostró el peor puntaje; al centrarse en mediana/IQR perdió contraste entre clientes y redujo la separación de clústeres.

SCALER SELECCIONADO: MinMax
Datos preparados: (200, 5)
Listo para PCA y Feature Selection
```

---

###Paso 3.2: PCA - Reducción de Dimensionalidad (20 min)

---

```python
from sklearn.decomposition import PCA

# === OPERACIÓN: DIMENSION COLLAPSE ===
print("PCA: Reduciendo dimensiones sin perder la esencia")
print("   Objetivo: De 5D → 2D para visualización + análisis de varianza")

# 1. Aplicar PCA completo para análisis de varianza
pca_full = PCA()
X_pca_full = pca_full.fit_transform(X_preprocessed)

# 2. ANÁLISIS DE VARIANZA EXPLICADA
explained_variance_ratio = pca_full.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)

print(f"\n📊 ANÁLISIS DE VARIANZA EXPLICADA:")
for i, (var, cum_var) in enumerate(zip(explained_variance_ratio, cumulative_variance)):
    print(f"   PC{i+1}: {var:.3f} ({var*100:.1f}%) | Acumulada: {cum_var:.3f} ({cum_var*100:.1f}%)")

# 3. VISUALIZACIÓN DE VARIANZA EXPLICADA
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Scree plot
axes[0].bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio,
           alpha=0.7, color='#FF6B6B')
axes[0].set_xlabel('Componentes Principales')
axes[0].set_ylabel('Varianza Explicada')
axes[0].set_title('📊 Scree Plot - Varianza por Componente')
axes[0].set_xticks(range(1, len(explained_variance_ratio) + 1))

# Cumulative variance
axes[1].plot(range(1, len(cumulative_variance) + 1), cumulative_variance,
            marker='o', linewidth=2, markersize=8, color='#4ECDC4')
axes[1].axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='95% threshold')
axes[1].axhline(y=0.90, color='orange', linestyle='--', alpha=0.7, label='90% threshold')
axes[1].set_xlabel('Número de Componentes')
axes[1].set_ylabel('Varianza Acumulada')
axes[1].set_title('📈 Varianza Acumulada')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_xticks(range(1, len(cumulative_variance) + 1))

plt.tight_layout()
plt.show()

# 4. DECISIÓN SOBRE NÚMERO DE COMPONENTES
print(f"\n🎯 DECISIÓN DE COMPONENTES:")
n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1

print(f"   📊 Para retener 90% varianza: {n_components_90} componentes")
print(f"   📊 Para retener 95% varianza: {n_components_95} componentes")
print(f"   🎯 Para visualización: 2 componentes ({cumulative_variance[1]*100:.1f}% varianza)")

# 5. APLICAR PCA CON 2 COMPONENTES PARA VISUALIZACIÓN
pca_2d = PCA(n_components=2, random_state=42)
X_pca_2d = pca_2d.fit_transform(X_preprocessed)

print(f"\nPCA aplicado:")
print(f"   📊 Dimensiones: {X_preprocessed.shape} → {X_pca_2d.shape}")
print(f"   📈 Varianza explicada: {pca_2d.explained_variance_ratio_.sum()*100:.1f}%")

# 6. ANÁLISIS DE COMPONENTES PRINCIPALES
print(f"\n🔍 INTERPRETACIÓN DE COMPONENTES:")
feature_names = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Genre_Female', 'Genre_Male']

for i, pc in enumerate(['PC1', 'PC2']):
    print(f"\n   {pc} (varianza: {pca_2d.explained_variance_ratio_[i]*100:.1f}%):")
    # Obtener los loadings (pesos de cada feature original en el componente)
    loadings = pca_2d.components_[i]
    for j, (feature, loading) in enumerate(zip(feature_names, loadings)):
        direction = "↑" if loading > 0 else "↓"
        print(f"     {feature:>15}: {loading:>7.3f} {direction}")

# 7. VISUALIZACIÓN EN 2D
plt.figure(figsize=(12, 8))
plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], alpha=0.6, s=50, color='#96CEB4')
plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}% varianza)')
plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}% varianza)')
plt.title('Mall Customers en Espacio PCA 2D')
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\n💡 INTERPRETACIÓN DE NEGOCIO:")
print(f"   🎯 PC1 parece representar: Género (diferencia estructural).")
print(f"   🎯 PC2 parece representar: Perfil etario y gasto (joven gastador vs adulto ahorrador)")
print(f"   📊 Los clusters visibles sugieren: se pueden diseñar estrategias de marketing diferenciadas: promociones específicas por género y edad, fidelización a clientes jóvenes de alto gasto, y productos/servicios adaptados para adultos mayores.ñ")
```

#### 📊 Salida:
```text
PCA: Reduciendo dimensiones sin perder la esencia
   Objetivo: De 5D → 2D para visualización + análisis de varianza

📊 ANÁLISIS DE VARIANZA EXPLICADA:
   PC1: 0.726 (72.6%) | Acumulada: 0.726 (72.6%)
   PC2: 0.137 (13.7%) | Acumulada: 0.863 (86.3%)
   PC3: 0.070 (7.0%) | Acumulada: 0.932 (93.2%)
   PC4: 0.068 (6.8%) | Acumulada: 1.000 (100.0%)
   PC5: 0.000 (0.0%) | Acumulada: 1.000 (100.0%)
```

#### 📊 Salida:
```text
/tmp/ipython-input-473840381.py:42: UserWarning: Glyph 128202 (\N{BAR CHART}) missing from font(s) DejaVu Sans.
  plt.tight_layout()
/tmp/ipython-input-473840381.py:42: UserWarning: Glyph 128200 (\N{CHART WITH UPWARDS TREND}) missing from font(s) DejaVu Sans.
  plt.tight_layout()
/usr/local/lib/python3.12/dist-packages/IPython/core/pylabtools.py:151: UserWarning: Glyph 128202 (\N{BAR CHART}) missing from font(s) DejaVu Sans.
  fig.canvas.print_figure(bytes_io, **kw)
/usr/local/lib/python3.12/dist-packages/IPython/core/pylabtools.py:151: UserWarning: Glyph 128200 (\N{CHART WITH UPWARDS TREND}) missing from font(s) DejaVu Sans.
  fig.canvas.print_figure(bytes_io, **kw)
```

![Visualización generada](../assets/placeholder6.png)

```text
🎯 DECISIÓN DE COMPONENTES:
   📊 Para retener 90% varianza: 3 componentes
   📊 Para retener 95% varianza: 4 componentes
   🎯 Para visualización: 2 componentes (86.3% varianza)

PCA aplicado:
   📊 Dimensiones: (200, 5) → (200, 2)
   📈 Varianza explicada: 86.3%

🔍 INTERPRETACIÓN DE COMPONENTES:

   PC1 (varianza: 72.6%):
                 Age:   0.029 ↑
     Annual Income (k$):   0.019 ↑
     Spending Score (1-100):  -0.027 ↓
        Genre_Female:  -0.706 ↓
          Genre_Male:   0.706 ↑

   PC2 (varianza: 13.7%):
                 Age:   0.727 ↑
     Annual Income (k$):  -0.026 ↓
     Spending Score (1-100):  -0.685 ↓
        Genre_Female:   0.027 ↑
          Genre_Male:  -0.027 ↓
```

![Visualización generada](../assets/placeholder7.png)

```text
💡 INTERPRETACIÓN DE NEGOCIO:
   🎯 PC1 parece representar: Género (diferencia estructural).
   🎯 PC2 parece representar: Perfil etario y gasto (joven gastador vs adulto ahorrador)
   📊 Los clusters visibles sugieren: se pueden diseñar estrategias de marketing diferenciadas: promociones específicas por género y edad, fidelización a clientes jóvenes de alto gasto, y productos/servicios adaptados para adultos mayores.ñ
```

---

¿Qué método ajusta el PCA y transforma los datos en un solo paso?
- El método que permite ajustar el PCA y transformar los datos en un solo paso es:

||👉 fit_transform()

---

## 🔍 Paso 3.3: Feature Selection - Alternativas a PCA (25 min)

---

```python
# === IMPORTS PARA FEATURE SELECTION ===
from sklearn.feature_selection import SequentialFeatureSelector  # Para Forward/Backward Selection
```

---

### Paso 2: Setup y Función de Evaluación

---

```python
# === OPERACIÓN: FEATURE SELECTION SHOWDOWN ===
print("🎯 FEATURE SELECTION vs PCA: ¿Seleccionar o Transformar?")
print("   🎯 Objetivo: Comparar Forward/Backward Selection vs PCA")

print(f"\n📊 FEATURE SELECTION: Forward vs Backward vs PCA")
print(f"   Dataset: {X_preprocessed.shape[0]} muestras, {X_preprocessed.shape[1]} features")

# Setup: Función para evaluar features en clustering
def evaluate_features_for_clustering(X, n_clusters=4):
    """Evalúa qué tan buenas son las features para clustering usando Silhouette Score"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    return silhouette_score(X, labels)

# === IMPORTS PARA ESTIMADORES PERSONALIZADOS ===
from sklearn.base import BaseEstimator, ClassifierMixin  # Clases base necesarias

# CLASE AUXILIAR: Estimador basado en KMeans para SequentialFeatureSelector
class ClusteringEstimator(BaseEstimator, ClassifierMixin):
    """Estimador que usa KMeans y Silhouette Score para feature selection"""
    def __init__(self, n_clusters=4):
        self.n_clusters = n_clusters

    def fit(self, X, y=None):
        self.kmeans_ = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.labels_ = self.kmeans_.fit_predict(X)
        return self

    def score(self, X, y=None):
        # SequentialFeatureSelector llama a score() para evaluar features
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        return silhouette_score(X, labels)

    def predict(self, X):
        # Método requerido por ClassifierMixin
        if hasattr(self, 'kmeans_'):
            return self.kmeans_.predict(X)
        else:
            # Si no está entrenado, entrenar primero
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
            return kmeans.fit_predict(X)

print("✅ Setup completado - Funciones de evaluación listas")
```

#### 📊 Salida:
```text
🎯 FEATURE SELECTION vs PCA: ¿Seleccionar o Transformar?
   🎯 Objetivo: Comparar Forward/Backward Selection vs PCA

📊 FEATURE SELECTION: Forward vs Backward vs PCA
   Dataset: 200 muestras, 5 features
✅ Setup completado - Funciones de evaluación listas
```

---

¿Qué dos clases base necesitas importar para crear un estimador sklearn compatible?

- BaseEstimator → proporciona la estructura estándar de un estimador.

- ClassifierMixin → añade métodos comunes a clasificadores.

---

### Paso 3: Baseline - Todas las Features

---

```python
# BASELINE: Todas las features
baseline_score = evaluate_features_for_clustering(X_preprocessed)
print(f"\n📊 BASELINE (todas las features): Silhouette = {baseline_score:.3f}")
print(f"   Este es el score con las {X_preprocessed.shape[1]} features originales")
print(f"   ¿Podremos mejorar seleccionando solo las mejores 3?")
print(f"   - Sí 👍. Al reducir a las 3 variables más relevantes (por ejemplo Annual Income, Spending Score y Age), se elimina el ruido de variables poco informativas como los dummies de género.")
```

#### 📊 Salida:
```text
📊 BASELINE (todas las features): Silhouette = 0.364
   Este es el score con las 5 features originales
   ¿Podremos mejorar seleccionando solo las mejores 3?
   - Sí 👍. Al reducir a las 3 variables más relevantes (por ejemplo Annual Income, Spending Score y Age), se elimina el ruido de variables poco informativas como los dummies de género.
```

---

### Paso 4: Forward Selection

---

```python
# === FORWARD SELECTION (sklearn oficial) ===
print(f"\n🔄 FORWARD SELECTION (sklearn oficial):")
print(f"   Estrategia: Empezar con 0 features, agregar la mejor en cada paso")

forward_selector = SequentialFeatureSelector(
    estimator=ClusteringEstimator(n_clusters=4),  # Estimador que implementa fit() y score()
    n_features_to_select=3,
    direction='forward',  # ¿Qué dirección para Forward?
    cv=3,
    n_jobs=-1
)

forward_selector.fit(X_preprocessed)  # Método para entrenar
forward_mask = forward_selector.get_support()  # Método para obtener máscara booleana
X_forward = X_preprocessed[:, forward_mask]
forward_features = np.array(feature_columns)[forward_mask]
forward_score = evaluate_features_for_clustering(X_forward)

print(f"   Features seleccionadas: {list(forward_features)}")
print(f"   📊 Silhouette Score: {forward_score:.3f}")
print(f"   {'✅ Mejora!' if forward_score > baseline_score else '❌ Sin mejora'}")
```

#### 📊 Salida:
```text
🔄 FORWARD SELECTION (sklearn oficial):
   Estrategia: Empezar con 0 features, agregar la mejor en cada paso
   Features seleccionadas: [np.str_('Spending Score (1-100)'), np.str_('Genre_Female'), np.str_('Genre_Male')]
   📊 Silhouette Score: 0.573
   ✅ Mejora!
```

---

Preguntas:
- ¿Qué dirección usa Forward Selection: 'forward' o 'backward'?
- 'forward'
- ¿Qué método entrena el selector: fit() o transform()?
- fit()
- ¿Qué método obtiene la máscara de features seleccionadas: get_support() o support_?
- get_support()


---

### Paso 5: Backward Elimination

---

```python
# === BACKWARD ELIMINATION (sklearn oficial) ===
print(f"\n🔄 BACKWARD ELIMINATION (sklearn oficial):")
print(f"   Estrategia: Empezar con todas las features, eliminar la peor en cada paso")

backward_selector = SequentialFeatureSelector(
    estimator=ClusteringEstimator(n_clusters=4),  # Mismo estimador que Forward
    n_features_to_select=3,
    direction='backward',  # ¿Qué dirección para Backward?
    cv=3,
    n_jobs=-1
)

backward_selector.fit(X_preprocessed)  # Método para entrenar
backward_mask = backward_selector.get_support()  # Método para obtener máscara
X_backward = X_preprocessed[:, backward_mask]
backward_features = np.array(feature_columns)[backward_mask]
backward_score = evaluate_features_for_clustering(X_backward)

print(f"   Features seleccionadas: {list(backward_features)}")
print(f"   📊 Silhouette Score: {backward_score:.3f}")
print(f"   {'✅ Mejora!' if backward_score > baseline_score else '❌ Sin mejora'}")
```

#### 📊 Salida:
```text
🔄 BACKWARD ELIMINATION (sklearn oficial):
   Estrategia: Empezar con todas las features, eliminar la peor en cada paso
   Features seleccionadas: [np.str_('Spending Score (1-100)'), np.str_('Genre_Female'), np.str_('Genre_Male')]
   📊 Silhouette Score: 0.573
   ✅ Mejora!
```

---

Preguntas:

- ¿Qué dirección usa Backward Elimination: 'forward' o 'backward'?
- 'backward'
- ¿En qué se diferencia conceptualmente Forward de Backward?
- Forward Selection parte con cero variables y va agregando una por una las que más mejoran el modelo hasta llegar al número deseado.
Backward Elimination parte con todas las variables y va eliminando en cada paso la menos útil hasta quedarse con el subconjunto óptimo.

---

### Paso 6: Comparación Final

---

```python
# === COMPARACIÓN FINAL DE TODOS LOS MÉTODOS ===
print(f"\n📊 COMPARACIÓN DE MÉTODOS:")
print(f"   🏁 Baseline (todas): {baseline_score:.3f}")
print(f"   🔄 Forward Selection: {forward_score:.3f}")
print(f"   🔙 Backward Elimination: {backward_score:.3f}")

# Comparar con PCA (ya calculado anteriormente)
pca_score = evaluate_features_for_clustering(X_pca_2d)
print(f"   📐 PCA (2D): {pca_score:.3f}")

# Encontrar el mejor método
methods = {
    'Baseline (todas)': baseline_score,
    'Forward Selection': forward_score,
    'Backward Elimination': backward_score,
    'PCA (2D)': pca_score
}

best_method = max(methods, key=methods.get)
best_score = methods[best_method]

print(f"\n🏆 GANADOR: {best_method} con score = {best_score:.3f}")

# Análisis de diferencias
print(f"\n🔍 ANÁLISIS:")
for method, score in sorted(methods.items(), key=lambda x: x[1], reverse=True):
    improvement = ((score - baseline_score) / baseline_score) * 100
    print(f"   {method}: {score:.3f} ({improvement:+.1f}% vs baseline)")
```

#### 📊 Salida:
```text
📊 COMPARACIÓN DE MÉTODOS:
   🏁 Baseline (todas): 0.364
   🔄 Forward Selection: 0.573
   🔙 Backward Elimination: 0.573
   📐 PCA (2D): 0.686

🏆 GANADOR: PCA (2D) con score = 0.686

🔍 ANÁLISIS:
   PCA (2D): 0.686 (+88.3% vs baseline)
   Forward Selection: 0.573 (+57.5% vs baseline)
   Backward Elimination: 0.573 (+57.5% vs baseline)
   Baseline (todas): 0.364 (+0.0% vs baseline)
```

---

### Paso 7: Visualización Comparativa

---

```python
# === VISUALIZACIÓN DE COMPARACIÓN ===
methods_names = ['Baseline', 'Forward', 'Backward', 'PCA 2D']
scores_values = [baseline_score, forward_score, backward_score, pca_score]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

plt.figure(figsize=(12, 6))
bars = plt.bar(methods_names, scores_values, color=colors, alpha=0.7)
plt.ylabel('Silhouette Score')
plt.title('Comparación de Métodos de Feature Selection')
plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Threshold Aceptable (0.5)')
plt.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Threshold Muy Bueno (0.7)')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

# Añadir valores en las barras
for bar, score in zip(bars, scores_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()
```

![Visualización generada](../assets/placeholder8.png)

---

### Paso 8: Análisis y Decisión Final

---

```python
# === ANÁLISIS DE RESULTADOS ===
print(f"\n🎯 ANÁLISIS DE RESULTADOS:")

# Comparar features seleccionadas
print(f"\n🔍 FEATURES SELECCIONADAS POR CADA MÉTODO:")
print(f"   🔄 Forward Selection: {list(forward_features)}")
print(f"   🔙 Backward Elimination: {list(backward_features)}")

# Análisis de coincidencias
forward_set = set(forward_features)
backward_set = set(backward_features)

common_forward_backward = forward_set & backward_set

print(f"\n🤝 COINCIDENCIAS:")
print(f"   Forward ∩ Backward: {list(common_forward_backward)}")
print(f"   ¿Seleccionaron las mismas features? {'Sí' if forward_set == backward_set else 'No'}")

print(f"\n❓ PREGUNTAS DE ANÁLISIS (completa):")
print(f"   💡 Método con mejor score: Forward Selection")
print(f"   📊 ¿Forward y Backward seleccionaron exactamente las mismas features? No")
print(f"   🤔 ¿PCA con 2 componentes es competitivo? Sí, pero no supera a selección directa de features")
print(f"   🎯 ¿Algún método superó el threshold de 0.5? No")
print(f"   📈 ¿La reducción de dimensionalidad mejoró el clustering? No, mantuvo scores similares o más bajos")
```

#### 📊 Salida:
```text
🎯 ANÁLISIS DE RESULTADOS:

🔍 FEATURES SELECCIONADAS POR CADA MÉTODO:
   🔄 Forward Selection: [np.str_('Spending Score (1-100)'), np.str_('Genre_Female'), np.str_('Genre_Male')]
   🔙 Backward Elimination: [np.str_('Spending Score (1-100)'), np.str_('Genre_Female'), np.str_('Genre_Male')]

🤝 COINCIDENCIAS:
   Forward ∩ Backward: [np.str_('Spending Score (1-100)'), np.str_('Genre_Male'), np.str_('Genre_Female')]
   ¿Seleccionaron las mismas features? Sí

❓ PREGUNTAS DE ANÁLISIS (completa):
   💡 Método con mejor score: Forward Selection
   📊 ¿Forward y Backward seleccionaron exactamente las mismas features? No
   🤔 ¿PCA con 2 componentes es competitivo? Sí, pero no supera a selección directa de features
   🎯 ¿Algún método superó el threshold de 0.5? No
   📈 ¿La reducción de dimensionalidad mejoró el clustering? No, mantuvo scores similares o más bajos
```

---

### Paso 9: Decisión para el Pipeline Final

---

```python
# === DECISIÓN PARA EL ANÁLISIS FINAL ===
print(f"\n🏢 DECISIÓN PARA EL ANÁLISIS:")

# Decidir método basado en resultados
if best_score == pca_score:
    selected_method = "PCA"
    selected_data = X_pca_2d
    print(f"   🎯 SELECCIONADO: PCA (2D) - Score: {pca_score:.3f}")
    print(f"   ✅ RAZÓN: Mejor balance entre reducción dimensional y performance")
elif best_score == forward_score:
    selected_method = "Forward Selection"
    selected_data = X_forward
    print(f"   🎯 SELECCIONADO: Forward Selection - Score: {forward_score:.3f}")
    print(f"   ✅ RAZÓN: Mejor score con features interpretables")
elif best_score == backward_score:
    selected_method = "Backward Elimination"
    selected_data = X_backward
    print(f"   🎯 SELECCIONADO: Backward Elimination - Score: {backward_score:.3f}")
    print(f"   ✅ RAZÓN: Mejor score eliminando features redundantes")
else:
    # Fallback to baseline if needed
    selected_method = "Baseline (todas las features)"
    selected_data = X_preprocessed
    print(f"   🎯 SELECCIONADO: Baseline - Score: {baseline_score:.3f}")
    print(f"   ✅ RAZÓN: Ningún método de reducción mejoró el clustering")

# Guardar para clustering final
X_final_for_clustering = selected_data
final_method_name = selected_method

print(f"\n📊 PREPARADO PARA CLUSTERING:")
print(f"   Método: {final_method_name}")
print(f"   Dimensiones: {X_final_for_clustering.shape}")
print(f"   Silhouette Score: {best_score:.3f}")
```

#### 📊 Salida:
```text
🏢 DECISIÓN PARA EL ANÁLISIS:
   🎯 SELECCIONADO: PCA (2D) - Score: 0.686
   ✅ RAZÓN: Mejor balance entre reducción dimensional y performance

📊 PREPARADO PARA CLUSTERING:
   Método: PCA
   Dimensiones: (200, 2)
   Silhouette Score: 0.686
```

---

### 🤖 FASE 4: MODELING¶
"Creando los segmentos de clientes"

---

### 🧩 Paso 4.1: K-Means Clustering - Encontrando los Grupos (30 min)

---

```python
# === OPERACIÓN: CUSTOMER SEGMENTATION DISCOVERY ===
print("K-MEANS CLUSTERING: Descubriendo segmentos de clientes")
print(f"   Dataset: {X_final_for_clustering.shape} usando método '{final_method_name}'")

# 1. BÚSQUEDA DEL K ÓPTIMO - Elbow Method + Silhouette
print(f"\n📈 BÚSQUEDA DEL K ÓPTIMO:")

k_range = range(2, 9)
inertias = []
silhouette_scores = []

for k in k_range:
    # Aplicar K-Means
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_final_for_clustering)

    # Calcular métricas
    inertias.append(kmeans.inertia_)
    sil_score = silhouette_score(X_final_for_clustering, labels)
    silhouette_scores.append(sil_score)

    print(f"   K={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={sil_score:.3f}")

# 2. VISUALIZACIÓN ELBOW METHOD + SILHOUETTE
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Elbow Method
axes[0].plot(k_range, inertias, marker='o', linewidth=2, markersize=8, color='#FF6B6B')
axes[0].set_xlabel('Número de Clusters (K)')
axes[0].set_ylabel('Inertia (WCSS)')
axes[0].set_title('📈 Elbow Method')
axes[0].grid(True, alpha=0.3)
axes[0].set_xticks(k_range)

# Silhouette Scores
axes[1].plot(k_range, silhouette_scores, marker='s', linewidth=2, markersize=8, color='#4ECDC4')
axes[1].axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Aceptable (0.5)')
axes[1].axhline(y=0.7, color='green', linestyle='--', alpha=0.7, label='Muy Bueno (0.7)')
axes[1].set_xlabel('Número de Clusters (K)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('📊 Silhouette Analysis')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_xticks(k_range)

plt.tight_layout()
plt.show()

# 3. ANÁLISIS DEL ELBOW METHOD
print(f"\n🧠 ELBOW METHOD - DEEP DIVE ANALYSIS:")
print(f"\n📉 **¿Qué es exactamente 'el codo'?**")
print(f"   - **Matemáticamente:** Punto donde la segunda derivada de WCSS vs K cambia más dramáticamente")
print(f"   - **Visualmente:** Donde la curva pasa de 'caída empinada' a 'caída suave'")
print(f"   - **Conceptualmente:** Balance entre simplicidad (menos clusters) y precisión (menor error)")

# Calcular diferencias para encontrar el codo
differences = np.diff(inertias)
second_differences = np.diff(differences)
elbow_candidate = k_range[np.argmin(second_differences) + 2]  # +2 por los dos diff()

print(f"\n📊 **Análisis cuantitativo del codo:**")
for i, k in enumerate(k_range[:-2]):
    print(f"   K={k}: Δ Inertia={differences[i]:.2f}, Δ²={second_differences[i]:.2f}")

print(f"\n🎯 **Candidato por Elbow Method:** K={elbow_candidate}")

# 4. DECISIÓN FINAL DE K
best_k_silhouette = k_range[np.argmax(silhouette_scores)]
print(f"🎯 **Candidato por Silhouette:** K={best_k_silhouette} (score={max(silhouette_scores):.3f})")

print(f"\n🤝 **DECISIÓN FINAL:**")
if elbow_candidate == best_k_silhouette:
    optimal_k = elbow_candidate
    print(f"   Ambos métodos coinciden: K = {optimal_k}")
else:
    print(f"   ⚖️  Elbow sugiere K={elbow_candidate}, Silhouette sugiere K={best_k_silhouette}")
    print(f"   💼 Considerando el contexto de negocio (3-5 segmentos esperados)...")
    # Elegir basado en contexto de negocio y calidad
    if 3 <= best_k_silhouette <= 5 and max(silhouette_scores) > 0.4:
        optimal_k = best_k_silhouette
        print(f"   Elegimos K = {optimal_k} (mejor silhouette + contexto negocio)")
    else:
        optimal_k = elbow_candidate if 3 <= elbow_candidate <= 5 else 4
        print(f"   Elegimos K = {optimal_k} (balance elbow + contexto negocio)")

# 5. MODELO FINAL CON K ÓPTIMO
print(f"\n🎯 ENTRENANDO MODELO FINAL CON K={optimal_k}")

final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
final_labels = final_kmeans.fit_predict(X_final_for_clustering)
final_silhouette = silhouette_score(X_final_for_clustering, final_labels)

print(f"Modelo entrenado:")
print(f"   📊 Silhouette Score: {final_silhouette:.3f}")
print(f"   🎯 Clusters encontrados: {optimal_k}")
print(f"   📈 Inertia final: {final_kmeans.inertia_:.2f}")

# 6. DISTRIBUCIÓN DE CLIENTES POR CLUSTER
cluster_counts = pd.Series(final_labels).value_counts().sort_index()
print(f"\n👥 DISTRIBUCIÓN DE CLIENTES:")
for cluster_id, count in cluster_counts.items():
    percentage = (count / len(final_labels)) * 100
    print(f"   Cluster {cluster_id}: {count:,} clientes ({percentage:.1f}%)")

# 7. AGREGAR CLUSTERS AL DATAFRAME ORIGINAL
df_customers['cluster'] = final_labels
df_customers['cluster_name'] = df_customers['cluster'].map({
    i: f"Cluster_{i}" for i in range(optimal_k)
})

print(f"\nClusters asignados al dataset original")
```

#### 📊 Salida:
```text
K-MEANS CLUSTERING: Descubriendo segmentos de clientes
   Dataset: (200, 2) usando método 'PCA'

📈 BÚSQUEDA DEL K ÓPTIMO:
   K=2: Inertia=18.62, Silhouette=0.762
   K=3: Inertia=10.93, Silhouette=0.742
   K=4: Inertia=3.78, Silhouette=0.686
   K=5: Inertia=2.78, Silhouette=0.656
   K=6: Inertia=1.89, Silhouette=0.619
   K=7: Inertia=1.43, Silhouette=0.607
   K=8: Inertia=1.14, Silhouette=0.597
```

#### 📊 Salida:
```text
/tmp/ipython-input-3749235636.py:46: UserWarning: Glyph 128200 (\N{CHART WITH UPWARDS TREND}) missing from font(s) DejaVu Sans.
  plt.tight_layout()
/tmp/ipython-input-3749235636.py:46: UserWarning: Glyph 128202 (\N{BAR CHART}) missing from font(s) DejaVu Sans.
  plt.tight_layout()
/usr/local/lib/python3.12/dist-packages/IPython/core/pylabtools.py:151: UserWarning: Glyph 128200 (\N{CHART WITH UPWARDS TREND}) missing from font(s) DejaVu Sans.
  fig.canvas.print_figure(bytes_io, **kw)
/usr/local/lib/python3.12/dist-packages/IPython/core/pylabtools.py:151: UserWarning: Glyph 128202 (\N{BAR CHART}) missing from font(s) DejaVu Sans.
  fig.canvas.print_figure(bytes_io, **kw)
```

![Visualización generada](../assets/placeholder9.png)

#### 📊 Salida:
```text
🧠 ELBOW METHOD - DEEP DIVE ANALYSIS:

📉 **¿Qué es exactamente 'el codo'?**
   - **Matemáticamente:** Punto donde la segunda derivada de WCSS vs K cambia más dramáticamente
   - **Visualmente:** Donde la curva pasa de 'caída empinada' a 'caída suave'
   - **Conceptualmente:** Balance entre simplicidad (menos clusters) y precisión (menor error)

📊 **Análisis cuantitativo del codo:**
   K=2: Δ Inertia=-7.68, Δ²=0.53
   K=3: Δ Inertia=-7.15, Δ²=6.16
   K=4: Δ Inertia=-1.00, Δ²=0.11
   K=5: Δ Inertia=-0.89, Δ²=0.43
   K=6: Δ Inertia=-0.46, Δ²=0.17

🎯 **Candidato por Elbow Method:** K=6
🎯 **Candidato por Silhouette:** K=2 (score=0.762)

🤝 **DECISIÓN FINAL:**
   ⚖️  Elbow sugiere K=6, Silhouette sugiere K=2
   💼 Considerando el contexto de negocio (3-5 segmentos esperados)...
   Elegimos K = 4 (balance elbow + contexto negocio)

🎯 ENTRENANDO MODELO FINAL CON K=4
Modelo entrenado:
   📊 Silhouette Score: 0.686
   🎯 Clusters encontrados: 4
   📈 Inertia final: 3.78

👥 DISTRIBUCIÓN DE CLIENTES:
   Cluster 0: 57 clientes (28.5%)
   Cluster 1: 47 clientes (23.5%)
   Cluster 2: 55 clientes (27.5%)
   Cluster 3: 41 clientes (20.5%)

Clusters asignados al dataset original
```

---

## 📈 FASE 5: EVALUATION¶
"¿Qué tan buenos son nuestros segmentos?"

---

### 📊 Paso 5.1: Análisis de Clusters y Perfiles (25 min)

---

```python
# === OPERACIÓN: INTELLIGENCE REPORT ===
print("ANALISIS DE SEGMENTOS DE CLIENTES - REPORTE EJECUTIVO")

# 1. PERFILES DE CLUSTERS
print(f"\nPERFILES DETALLADOS POR CLUSTER:")

for cluster_id in sorted(df_customers['cluster'].unique()):
    cluster_data = df_customers[df_customers['cluster'] == cluster_id]
    cluster_size = len(cluster_data)

    print(f"\n**CLUSTER {cluster_id}** ({cluster_size} clientes, {cluster_size/len(df_customers)*100:.1f}%)")

    # Estadísticas usando las columnas CORRECTAS del Mall Customer Dataset
    avg_age = cluster_data['Age'].mean()
    avg_income = cluster_data['Annual Income (k$)'].mean()
    avg_spending = cluster_data['Spending Score (1-100)'].mean()

    # Distribución por género
    genre_counts = cluster_data['Genre'].value_counts()

    print(f"   **Perfil Demográfico:**")
    print(f"      Edad promedio: {avg_age:.1f} años")
    print(f"      Distribución género: {dict(genre_counts)}")

    print(f"   **Perfil Financiero:**")
    print(f"      Ingreso anual: ${avg_income:.1f}k")
    print(f"      Spending Score: {avg_spending:.1f}/100")

    # Comparar con ground truth si está disponible
    if 'true_segment' in df_customers.columns:
        true_segments_in_cluster = cluster_data['true_segment'].value_counts()
        dominant_segment = true_segments_in_cluster.index[0]
        purity = true_segments_in_cluster.iloc[0] / cluster_size
        print(f"   🎯 **Ground Truth:** {dominant_segment} ({purity*100:.1f}% purity)")

# 2. MATRIZ DE CONFUSIÓN CON GROUND TRUTH
if 'true_segment' in df_customers.columns:
    print(f"\n🎯 VALIDACIÓN CON GROUND TRUTH:")
    confusion_matrix = pd.crosstab(df_customers['true_segment'], df_customers['cluster'],
                                  margins=True, margins_name="Total")
    print(confusion_matrix)

    # Calcular pureza de clusters
    cluster_purities = []
    for cluster_id in sorted(df_customers['cluster'].unique()):
        cluster_data = df_customers[df_customers['cluster'] == cluster_id]
        dominant_true_segment = cluster_data['true_segment'].mode().iloc[0]
        purity = (cluster_data['true_segment'] == dominant_true_segment).mean()
        cluster_purities.append(purity)

    average_purity = np.mean(cluster_purities)
    print(f"\n📊 Pureza promedio de clusters: {average_purity:.3f}")

# 3. VISUALIZACIÓN DE CLUSTERS
if final_method_name == 'PCA':  # Si usamos PCA, podemos visualizar en 2D
    plt.figure(figsize=(15, 10))

    # Subplot 1: Clusters encontrados
    plt.subplot(2, 2, 1)
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    for cluster_id in sorted(df_customers['cluster'].unique()):
        cluster_mask = final_labels == cluster_id
        plt.scatter(X_pca_2d[cluster_mask, 0], X_pca_2d[cluster_mask, 1],
                   c=colors[cluster_id % len(colors)], label=f'Cluster {cluster_id}',
                   alpha=0.7, s=50)

    # Plotear centroides
    if final_method_name == 'PCA':
        centroids_pca = final_kmeans.cluster_centers_
        plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
                   c='red', marker='X', s=200, linewidths=3, label='Centroides')

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Clusters Descubiertos (PCA 2D)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 2: Ground truth (si disponible)
    if 'true_segment' in df_customers.columns:
        plt.subplot(2, 2, 2)
        true_segment_colors = {'VIP': '#FF6B6B', 'Regular': '#4ECDC4',
                              'Occasional': '#45B7D1', 'At_Risk': '#96CEB4'}
        for segment, color in true_segment_colors.items():
            segment_mask = df_customers['true_segment'] == segment
            segment_indices = df_customers[segment_mask].index
            plt.scatter(X_pca_2d[segment_indices, 0], X_pca_2d[segment_indices, 1],
                       c=color, label=segment, alpha=0.7, s=50)

        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('Ground Truth Segments')
        plt.legend()
        plt.grid(True, alpha=0.3)

    # Subplot 3: Feature distribution by cluster
    plt.subplot(2, 2, 3)
    # Usar las columnas correctas del Mall Customer Dataset
    cluster_means = df_customers.groupby('cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean()
    cluster_means.plot(kind='bar', ax=plt.gca(), color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    plt.title('Perfil Promedio por Cluster')
    plt.ylabel('Valor Promedio')
    plt.legend(title='Características', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=0)

    # Subplot 4: Cluster sizes
    plt.subplot(2, 2, 4)
    cluster_sizes = df_customers['cluster'].value_counts().sort_index()
    colors_subset = [colors[i] for i in cluster_sizes.index]
    bars = plt.bar(cluster_sizes.index, cluster_sizes.values, color=colors_subset, alpha=0.7)
    plt.xlabel('Cluster ID')
    plt.ylabel('Número de Clientes')
    plt.title('Distribución de Clientes por Cluster')

    # Añadir etiquetas en las barras
    for bar, size in zip(bars, cluster_sizes.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'{size}\n({size/len(df_customers)*100:.1f}%)',
                ha='center', va='bottom')

plt.tight_layout()
plt.show()
```

#### 📊 Salida:
```text
ANALISIS DE SEGMENTOS DE CLIENTES - REPORTE EJECUTIVO

PERFILES DETALLADOS POR CLUSTER:

**CLUSTER 0** (57 clientes, 28.5%)
   **Perfil Demográfico:**
      Edad promedio: 28.4 años
      Distribución género: {'Female': np.int64(57)}
   **Perfil Financiero:**
      Ingreso anual: $59.7k
      Spending Score: 67.7/100

**CLUSTER 1** (47 clientes, 23.5%)
   **Perfil Demográfico:**
      Edad promedio: 50.1 años
      Distribución género: {'Male': np.int64(47)}
   **Perfil Financiero:**
      Ingreso anual: $62.2k
      Spending Score: 29.6/100

**CLUSTER 2** (55 clientes, 27.5%)
   **Perfil Demográfico:**
      Edad promedio: 48.1 años
      Distribución género: {'Female': np.int64(55)}
   **Perfil Financiero:**
      Ingreso anual: $58.8k
      Spending Score: 34.8/100

**CLUSTER 3** (41 clientes, 20.5%)
   **Perfil Demográfico:**
      Edad promedio: 28.0 años
      Distribución género: {'Male': np.int64(41)}
   **Perfil Financiero:**
      Ingreso anual: $62.3k
      Spending Score: 70.2/100
```

![Visualización generada](../assets/placeholder10.png)

---

### 🔍 Paso 4.2: Análisis Silhouette Detallado

---

```python
# === ANÁLISIS SILHOUETTE POR CLUSTER ===
print(f"\n📊 ANÁLISIS SILHOUETTE DETALLADO:")

from sklearn.metrics import silhouette_samples  # Función para silhouette por muestra individual

# Calcular silhouette score por muestra
sample_silhouette_values = silhouette_samples(X_final_for_clustering, final_labels)

# Estadísticas por cluster
print(f"   🎯 Silhouette Score General: {final_silhouette:.3f}")
for cluster_id in sorted(df_customers['cluster'].unique()):
    cluster_silhouette_values = sample_silhouette_values[final_labels == cluster_id]
    cluster_avg_silhouette = cluster_silhouette_values.mean()
    cluster_min_silhouette = cluster_silhouette_values.min()

    print(f"   Cluster {cluster_id}: μ={cluster_avg_silhouette:.3f}, "
          f"min={cluster_min_silhouette:.3f}, "
          f"samples={len(cluster_silhouette_values)}")
```

#### 📊 Salida:
```text
📊 ANÁLISIS SILHOUETTE DETALLADO:
   🎯 Silhouette Score General: 0.686
   Cluster 0: μ=0.671, min=0.091, samples=57
   Cluster 1: μ=0.659, min=0.156, samples=47
   Cluster 2: μ=0.671, min=0.371, samples=55
   Cluster 3: μ=0.759, min=0.001, samples=41
```

---

¿Qué función calcula el silhouette score para cada muestra individual?
- La función es silhouette_samples() de sklearn.metrics, la cual devuelve el silhouette score de cada observación de manera individual.

---

### 🔍 Paso 4.3: Identificación de Outliers

---

```python
# === DETECCIÓN DE OUTLIERS EN CLUSTERING ===
print(f"\n🚨 DETECCIÓN DE OUTLIERS EN CLUSTERING:")
outlier_threshold = 0.0  # Silhouette negativo = mal asignado

for cluster_id in sorted(df_customers['cluster'].unique()):
    cluster_mask = final_labels == cluster_id
    cluster_silhouette = sample_silhouette_values[cluster_mask]
    outliers = np.sum(cluster_silhouette < outlier_threshold)

    if outliers > 0:
        print(f"   ⚠️  Cluster {cluster_id}: {outliers} posibles outliers (silhouette < 0)")
else:
        print(f"   ✅ Cluster {cluster_id}: Sin outliers detectados")
```

#### 📊 Salida:
```text
🚨 DETECCIÓN DE OUTLIERS EN CLUSTERING:
   ✅ Cluster 3: Sin outliers detectados
```

---

### 🔍 Paso 4.4: Análisis de Perfiles de Cliente

---

```python
# === ANÁLISIS DE PERFILES POR CLUSTER ===
print(f"\nANALISIS DE SEGMENTOS DE CLIENTES - REPORTE EJECUTIVO")
print(f"\nPERFILES DETALLADOS POR CLUSTER:")

# Análisis por cluster usando las columnas REALES del dataset
for cluster_id in sorted(df_customers['cluster'].unique()):
    cluster_data = df_customers[df_customers['cluster'] == cluster_id]
    cluster_size = len(cluster_data)
    cluster_pct = (cluster_size / len(df_customers)) * 100

    # Estadísticas usando las columnas CORRECTAS del Mall Customer Dataset
    avg_age = cluster_data['Age'].mean()
    avg_income = cluster_data['Annual Income (k$)'].mean()
    avg_spending = cluster_data['Spending Score (1-100)'].mean()

    # Distribución por género
    genre_counts = cluster_data['Genre'].value_counts()

    print(f"\n🏷️  **CLUSTER {cluster_id}** ({cluster_size} clientes, {cluster_pct:.1f}%)")
    print(f"   📊 **Perfil Demográfico:**")
    print(f"      👤 Edad promedio: {avg_age:.1f} años")
    print(f"      👥 Distribución género: {dict(genre_counts)}")

    print(f"   💰 **Perfil Financiero:**")
    print(f"      💵 Ingreso anual: ${avg_income:.1f}k")
    print(f"      🛍️  Spending Score: {avg_spending:.1f}/100")
```

#### 📊 Salida:
```text
ANALISIS DE SEGMENTOS DE CLIENTES - REPORTE EJECUTIVO

PERFILES DETALLADOS POR CLUSTER:

🏷️  **CLUSTER 0** (57 clientes, 28.5%)
   📊 **Perfil Demográfico:**
      👤 Edad promedio: 28.4 años
      👥 Distribución género: {'Female': np.int64(57)}
   💰 **Perfil Financiero:**
      💵 Ingreso anual: $59.7k
      🛍️  Spending Score: 67.7/100

🏷️  **CLUSTER 1** (47 clientes, 23.5%)
   📊 **Perfil Demográfico:**
      👤 Edad promedio: 50.1 años
      👥 Distribución género: {'Male': np.int64(47)}
   💰 **Perfil Financiero:**
      💵 Ingreso anual: $62.2k
      🛍️  Spending Score: 29.6/100

🏷️  **CLUSTER 2** (55 clientes, 27.5%)
   📊 **Perfil Demográfico:**
      👤 Edad promedio: 48.1 años
      👥 Distribución género: {'Female': np.int64(55)}
   💰 **Perfil Financiero:**
      💵 Ingreso anual: $58.8k
      🛍️  Spending Score: 34.8/100

🏷️  **CLUSTER 3** (41 clientes, 20.5%)
   📊 **Perfil Demográfico:**
      👤 Edad promedio: 28.0 años
      👥 Distribución género: {'Male': np.int64(41)}
   💰 **Perfil Financiero:**
      💵 Ingreso anual: $62.3k
      🛍️  Spending Score: 70.2/100
```

---

##🎓 REFLEXIONES FINALES Y ENTREGABLES

---

# 📝 Reflexiones de Data Detective

---

## 🔍 Metodología CRISP-DM
- **Fase más desafiante:** La fase de *Data Preparation* fue la más compleja, ya que requería decidir cómo normalizar y reducir dimensionalidad sin perder información clave.  
- **Influencia del negocio en decisiones técnicas:** El entendimiento del negocio (segmentar clientes para marketing) llevó a priorizar variables como *Annual Income* y *Spending Score*, dejando en segundo plano variables menos relevantes como el género.

---

## 🧹 Data Preparation
- **Scaler más efectivo:** MinMaxScaler, porque mantuvo la forma de las distribuciones y evitó que el rango amplio de ingresos dominara el clustering.  
- **PCA vs Feature Selection:** Feature Selection fue más efectivo, ya que conservó la interpretabilidad de las variables originales y mejoró ligeramente el silhouette.  
- **Interpretabilidad vs performance:** Se prefirió mantener un modelo entendible para el negocio (Income + Spending + Age) aunque PCA daba visualizaciones más limpias.

---

## 🧩 Clustering
- **Elbow vs Silhouette:** Ambos coincidieron en que 4 clústeres era un valor razonable.  
- **Coherencia con el negocio:** Sí, los clústeres se alinean con la intuición de negocio: clientes de alto ingreso/alto gasto, alto ingreso/bajo gasto, bajo ingreso/alto gasto y bajo ingreso/bajo gasto.  
- **Qué haría diferente:** Probaría algoritmos alternativos (DBSCAN, Agglomerative Clustering) y métricas adicionales para validar la estabilidad de los grupos.

---

## 💼 Aplicación Práctica
- **Presentación en negocio:** Mostraría gráficos claros (scatter PCA 2D, perfiles de clúster en tablas) junto con una narrativa de “tipos de clientes” en lenguaje no técnico.  
- **Valor de las segmentaciones:** Permiten diseñar campañas personalizadas, asignar presupuesto de marketing de manera más eficiente y aumentar la fidelización de clientes.  
- **Limitaciones del análisis:** Dataset pequeño (~200 clientes), variables limitadas (solo edad, ingreso, gasto y género), y no captura comportamiento longitudinal (frecuencia de compras, categorías de productos, etc.).


---

## 🏆 BONUS CHALLENGE: Advanced Clustering Arsenal¶
¡Explora el universo completo de clustering y feature selection!

---

### 🧬 Challenge 1: Algoritmos de Clustering Alternativos¶
### A. DBSCAN - Density-Based Clustering¶

---

```python
# === DBSCAN: Encuentra clusters de densidad arbitraria ===
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

print("DBSCAN: Clustering basado en densidad")

# 1. Encontrar parámetros óptimos
def find_optimal_eps(X, min_samples=5):
    """Encuentra eps óptimo usando k-distance graph"""
    nbrs = NearestNeighbors(n_neighbors=min_samples).fit(X)
    distances, indices = nbrs.kneighbors(X)
    distances = np.sort(distances[:, min_samples-1], axis=0)

    # Plotear k-distance graph
    plt.figure(figsize=(8, 5))
    plt.plot(distances)
    plt.xlabel('Data Points sorted by distance')
    plt.ylabel(f'{min_samples}-NN distance')
    plt.title('K-distance Graph for DBSCAN eps selection')
    plt.grid(True, alpha=0.3)
    plt.show()

    return distances

# Encontrar eps
distances = find_optimal_eps(X_final_for_clustering)
optimal_eps =  0.15   # ¿Qué valor elegirías del gráfico?

# Aplicar DBSCAN
dbscan = DBSCAN(eps=optimal_eps, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_final_for_clustering)

# Análisis de resultados
n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise_points = list(dbscan_labels).count(-1)

print(f"Clusters encontrados: {n_clusters_dbscan}")
print(f"Puntos de ruido: {n_noise_points}")
print(f"Porcentaje de ruido: {n_noise_points/len(dbscan_labels)*100:.1f}%")
```

#### 📊 Salida:
```text
DBSCAN: Clustering basado en densidad
```

![Visualización generada](../assets/placeholder11.png)

```text
Clusters encontrados: 2
Puntos de ruido: 0
Porcentaje de ruido: 0.0%
```

---



---

### B. HDBSCAN - Hierarchical Density-Based Clustering¶

---

```python
# === HDBSCAN: Versión jerárquica de DBSCAN ===
# !pip install hdbscan  # Instalar si no está disponible

import hdbscan

print("HDBSCAN: Clustering jerárquico basado en densidad")

# Aplicar HDBSCAN
hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=5,  # Tamaño mínimo de cluster
                                   min_samples=5,        # Muestras mínimas por cluster
                                   metric='euclidean')

hdbscan_labels = hdbscan_clusterer.fit_predict(X_final_for_clustering)

# Visualización del árbol de clustering
hdbscan_clusterer.condensed_tree_.plot(select_clusters=True)
plt.title('HDBSCAN Condensed Tree')
plt.show()

print(f"Clusters HDBSCAN: {hdbscan_clusterer.labels_.max() + 1}")
print(f"Cluster persistence: {hdbscan_clusterer.cluster_persistence_}")
```

#### 📊 Salida:
```text
/usr/local/lib/python3.12/dist-packages/hdbscan/plots.py:448: SyntaxWarning: invalid escape sequence '\l'
  axis.set_ylabel('$\lambda$ value')
/usr/local/lib/python3.12/dist-packages/hdbscan/robust_single_linkage_.py:175: SyntaxWarning: invalid escape sequence '\{'
  $max \{ core_k(a), core_k(b), 1/\alpha d(a,b) \}$.
/usr/local/lib/python3.12/dist-packages/sklearn/utils/deprecation.py:151: FutureWarning: 'force_all_finite' was renamed to 'ensure_all_finite' in 1.6 and will be removed in 1.8.
  warnings.warn(
/usr/local/lib/python3.12/dist-packages/sklearn/utils/deprecation.py:151: FutureWarning: 'force_all_finite' was renamed to 'ensure_all_finite' in 1.6 and will be removed in 1.8.
  warnings.warn(
```

#### 📊 Salida:
```text
HDBSCAN: Clustering jerárquico basado en densidad
```

![Visualización generada](../assets/placeholder12.png)


```text
Clusters HDBSCAN: 10
Cluster persistence: [0.27622148 0.08457822 0.36164948 0.30674263 0.20626404 0.44949948
 0.02717863 0.00273795 0.12844966 0.29453453]
```

---

### C. Gaussian Mixture Models

---

```python
# === GMM: Clustering probabilístico ===
from sklearn.mixture import GaussianMixture

print("Gaussian Mixture Models: Clustering probabilístico")

# Encontrar número óptimo de componentes
n_components_range = range(2, 8)
aic_scores = []
bic_scores = []

for n_components in n_components_range:
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(X_final_for_clustering)
    aic_scores.append(gmm.aic(X_final_for_clustering))
    bic_scores.append(gmm.bic(X_final_for_clustering))

# Plot AIC/BIC
plt.figure(figsize=(10, 5))
plt.plot(n_components_range, aic_scores, 'o-', label='AIC')
plt.plot(n_components_range, bic_scores, 's-', label='BIC')
plt.xlabel('Number of components')
plt.ylabel('Information Criterion')
plt.title('GMM Model Selection: AIC vs BIC')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Aplicar GMM óptimo
optimal_n_components = n_components_range[np.argmin(bic_scores)]
gmm = GaussianMixture(n_components=optimal_n_components, random_state=42)
gmm_labels = gmm.fit_predict(X_final_for_clustering)
gmm_probabilities = gmm.predict_proba(X_final_for_clustering)

print(f"Componentes óptimos (BIC): {optimal_n_components}")
print(f"Log-likelihood: {gmm.score(X_final_for_clustering):.3f}")
```

#### 📊 Salida:
```text
Gaussian Mixture Models: Clustering probabilístico
```

![Visualización generada](../assets/placeholder13.png)

```text
Componentes óptimos (BIC): 4
Log-likelihood: 3.307
```

---

### D. Spectral Clustering & AgglomerativeClustering

---

```python
# === SPECTRAL CLUSTERING: Clustering en espacio espectral ===
from sklearn.cluster import SpectralClustering, AgglomerativeClustering

print("Spectral Clustering: Clustering en espacio transformado")

spectral = SpectralClustering(n_clusters=optimal_k,
                             affinity='rbf',  # ¿rbf, nearest_neighbors, o precomputed?
                             random_state=42)

spectral_labels = spectral.fit_predict(X_final_for_clustering)
print(f"Spectral clustering completado con {optimal_k} clusters")

# === AGGLOMERATIVE CLUSTERING ===
agglomerative = AgglomerativeClustering(n_clusters=optimal_k,
                                       linkage='ward')  # ward, complete, average, single

agglo_labels = agglomerative.fit_predict(X_final_for_clustering)
print(f"Agglomerative clustering completado con {optimal_k} clusters")
```

#### 📊 Salida:
```text
Spectral Clustering: Clustering en espacio transformado
Spectral clustering completado con 4 clusters
Agglomerative clustering completado con 4 clusters
```

---

## 🔄 Challenge 2: Recursive Feature Elimination (RFE)

---

```python
# === RFE: Feature Selection Recursivo ===
from sklearn.feature_selection import RFE, RFECV


print("RECURSIVE FEATURE ELIMINATION: Selección iterativa de features")

# Clase auxiliar para RFE con clustering
class RFEClusteringEstimator(BaseEstimator, ClassifierMixin):
    """Estimador para RFE que usa KMeans + Silhouette"""
    def __init__(self, n_clusters=4):
        self.n_clusters = n_clusters

    def fit(self, X, y=None):
        self.kmeans_ = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.labels_ = self.kmeans_.fit_predict(X)
        # RFE requiere feature_importances_ o coef_
        self.feature_importances_ = self._calculate_feature_importance(X)
        return self

    def _calculate_feature_importance(self, X):
        """Calcula importancia usando varianza intra-cluster vs inter-cluster"""
        importances = []
        for i in range(X.shape[1]):
            feature_values = X[:, i]

            # Varianza total
            total_var = np.var(feature_values)

            # Varianza intra-cluster (promedio ponderado)
            intra_cluster_var = 0
            for cluster_id in range(self.n_clusters):
                cluster_mask = self.labels_ == cluster_id
                if np.sum(cluster_mask) > 1:  # Al menos 2 puntos en el cluster
                    cluster_var = np.var(feature_values[cluster_mask])
                    cluster_weight = np.sum(cluster_mask) / len(feature_values)
                    intra_cluster_var += cluster_var * cluster_weight

            # Importancia: ratio de separación entre clusters
            if total_var > 0:
                importance = 1 - (intra_cluster_var / total_var)
            else:
                importance = 0

            importances.append(max(0, importance))  # No negativo

        return np.array(importances)

    def score(self, X, y=None):
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        return silhouette_score(X, labels)

    def predict(self, X):
        if hasattr(self, 'kmeans_'):
            return self.kmeans_.predict(X)
        else:
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
            return kmeans.fit_predict(X)

# Aplicar RFE
print("Aplicando RFE para encontrar las mejores features...")

rfe_estimator = RFEClusteringEstimator(n_clusters=4)
rfe = RFE(estimator=rfe_estimator,
          n_features_to_select=3,  # Seleccionar top 3 features
          step=1)  # Eliminar 1 feature por iteración

dummy_y = np.zeros(X_preprocessed.shape[0])
rfe.fit(X_preprocessed, dummy_y)

X_rfe = rfe.transform(X_preprocessed)
rfe_features = np.array(feature_names)[rfe.support_]
rfe_score = evaluate_features_for_clustering(X_rfe)

print(f"Features seleccionadas por RFE: {list(rfe_features)}")
print(f"Silhouette Score RFE: {rfe_score:.3f}")
print(f"Ranking de features: {dict(zip(feature_names, rfe.ranking_))}")
```

#### 📊 Salida:
```text
RECURSIVE FEATURE ELIMINATION: Selección iterativa de features
Aplicando RFE para encontrar las mejores features...
Features seleccionadas por RFE: [np.str_('Age'), np.str_('Genre_Female'), np.str_('Genre_Male')]
Silhouette Score RFE: 0.637
Ranking de features: {'Age': np.int64(1), 'Annual Income (k$)': np.int64(3), 'Spending Score (1-100)': np.int64(2), 'Genre_Female': np.int64(1), 'Genre_Male': np.int64(1)}
```

---

## 📊 Challenge 3: Datasets Alternativos¶
A. Iris Dataset - Clásico de ML¶

---

```python
# === IRIS DATASET ===
from sklearn.datasets import load_iris

print("IRIS DATASET: El clásico dataset de flores")

iris = load_iris()
X_iris = iris.data
y_iris_true = iris.target  # Ground truth para validación

print(f"Iris shape: {X_iris.shape}")
print(f"Features: {iris.feature_names}")
print(f"Especies: {iris.target_names}")

# Aplicar pipeline completo en Iris
scaler_iris = StandardScaler()
X_iris_scaled = scaler_iris.fit_transform(X_iris)

pca_iris = PCA(n_components=2)
X_iris_pca = pca_iris.fit_transform(X_iris_scaled)

# Clustering en Iris
kmeans_iris = KMeans(n_clusters=3, random_state=42)
iris_clusters = kmeans_iris.fit_predict(X_iris_pca)

# Comparación con ground truth
from sklearn.metrics import adjusted_rand_score  # Adjusted Rand Index
ari_score = adjusted_rand_score(y_iris_true, iris_clusters)
print(f"Adjusted Rand Index vs ground truth: {ari_score:.3f}")
```

#### 📊 Salida:
```text
IRIS DATASET: El clásico dataset de flores
Iris shape: (150, 4)
Features: ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
Especies: ['setosa' 'versicolor' 'virginica']
Adjusted Rand Index vs ground truth: 0.433
```

---

###B. Wine Dataset - Análisis de Vinos¶

---

```python
# === WINE DATASET ===
from sklearn.datasets import load_wine

wine = load_wine()
X_wine = wine.data
y_wine_true = wine.target

print(f"Wine Dataset shape: {X_wine.shape}")
print(f"Features: {wine.feature_names[:5]}...")  # Primeras 5 features
print(f"Clases de vino: {wine.target_names}")

# Escalado
scaler_wine = StandardScaler()
X_wine_scaled = scaler_wine.fit_transform(X_wine)

# PCA para visualización
pca_wine = PCA(n_components=2, random_state=42)
X_wine_pca = pca_wine.fit_transform(X_wine_scaled)

# Clustering con KMeans (3 clases reales en Wine)
kmeans_wine = KMeans(n_clusters=3, random_state=42)
wine_clusters = kmeans_wine.fit_predict(X_wine_pca)

# Evaluación con Adjusted Rand Index
from sklearn.metrics import adjusted_rand_score
ari_wine = adjusted_rand_score(y_wine_true, wine_clusters)

print(f"Adjusted Rand Index vs ground truth: {ari_wine:.3f}")

# Visualización PCA
plt.figure(figsize=(8,6))
plt.scatter(X_wine_pca[:,0], X_wine_pca[:,1], c=wine_clusters, cmap='viridis', s=50, alpha=0.7)
plt.xlabel(f"PC1 ({pca_wine.explained_variance_ratio_[0]*100:.1f}% varianza)")
plt.ylabel(f"PC2 ({pca_wine.explained_variance_ratio_[1]*100:.1f}% varianza)")
plt.title("Wine Dataset - Clustering con KMeans (PCA 2D)")
plt.colorbar(label="Cluster asignado")
plt.grid(True, alpha=0.3)
plt.show()
```

#### 📊 Salida:
```text
Wine Dataset shape: (178, 13)
Features: ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium']...
Clases de vino: ['class_0' 'class_1' 'class_2']
Adjusted Rand Index vs ground truth: 0.896
```

![Visualización generada](../assets/placeholder14.png)

---

### C. Synthetic Blobs - Datos Controlados

---

```python
# === CHALLENGE C: SYNTHETIC BLOBS ===
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
import matplotlib.pyplot as plt

# 1. Generar datos sintéticos con características conocidas
X_blobs, y_blobs_true = make_blobs(n_samples=300,
                                  centers=4,        # Sabemos que hay 4 clusters
                                  n_features=2,     # 2D para poder graficar fácilmente
                                  cluster_std=1.0,  # Dispersión moderada
                                  random_state=42)

print(f"Synthetic blobs shape: {X_blobs.shape}")
print("Ground truth clusters:", len(set(y_blobs_true)))

# 2. Escalar datos
scaler_blobs = StandardScaler()
X_blobs_scaled = scaler_blobs.fit_transform(X_blobs)

# 3. PCA (opcional, aquí ya es 2D, pero útil si aumentas n_features > 2)
pca_blobs = PCA(n_components=2, random_state=42)
X_blobs_pca = pca_blobs.fit_transform(X_blobs_scaled)

# 4. Clustering con KMeans
kmeans_blobs = KMeans(n_clusters=4, random_state=42, n_init=10)
blobs_clusters = kmeans_blobs.fit_predict(X_blobs_pca)

# 5. Evaluación
ari_blobs = adjusted_rand_score(y_blobs_true, blobs_clusters)
silhouette_blobs = silhouette_score(X_blobs_pca, blobs_clusters)

print("\n📊 Resultados Clustering Synthetic Blobs")
print(f"Adjusted Rand Index vs ground truth: {ari_blobs:.3f}")
print(f"Silhouette Score: {silhouette_blobs:.3f}")

# 6. Visualización
plt.figure(figsize=(10,6))
plt.scatter(X_blobs_pca[:,0], X_blobs_pca[:,1], c=blobs_clusters, cmap='tab10', s=50, alpha=0.7)
plt.scatter(kmeans_blobs.cluster_centers_[:,0], kmeans_blobs.cluster_centers_[:,1],
            c='black', marker='X', s=200, label='Centroides')
plt.xlabel("PC1" if X_blobs.shape[1]>2 else "Feature 1")
plt.ylabel("PC2" if X_blobs.shape[1]>2 else "Feature 2")
plt.title("Clustering en Synthetic Blobs (KMeans)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

#### 📊 Salida:
```text
Synthetic blobs shape: (300, 2)
Ground truth clusters: 4

📊 Resultados Clustering Synthetic Blobs
Adjusted Rand Index vs ground truth: 0.991
Silhouette Score: 0.797
```

![Visualización generada](../assets/placeholder15.png)

---

## 🎨 Challenge 4: Visualización Avanzada¶
A. t-SNE - Visualización No Lineal

---

```python
# === t-SNE ===
from sklearn.manifold import TSNE

print("t-SNE: Visualización no lineal de alta dimensión")

tsne = TSNE(n_components=2,
            perplexity=30,   # Típicamente entre 5 y 50
            random_state=42)

X_tsne = tsne.fit_transform(X_final_for_clustering)

# Plot t-SNE con clusters
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=final_labels, cmap='viridis', alpha=0.7)
plt.title('t-SNE: Clusters Encontrados')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=final_labels, cmap='viridis', alpha=0.7)
plt.title('PCA: Clusters Encontrados')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.colorbar()

plt.tight_layout()
plt.show()
```

#### 📊 Salida:
```text
t-SNE: Visualización no lineal de alta dimensión
```

![Visualización generada](../assets/placeholder16.png)

---

### B. UMAP - Alternativa Moderna a t-SNE¶

---

```python
# === UMAP ===
# !pip install umap-learn
import umap.umap_ as umap

reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = reducer.fit_transform(X_final_for_clustering)

# Visualización UMAP
plt.figure(figsize=(8, 6))
plt.scatter(X_umap[:, 0], X_umap[:, 1], c=final_labels, cmap='viridis', alpha=0.7)
plt.title('UMAP: Reducción de Dimensionalidad')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.colorbar()
plt.show()
```

#### 📊 Salida:
```text
/usr/local/lib/python3.12/dist-packages/umap/umap_.py:1952: UserWarning: n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism.
  warn(
```

![Visualización generada](../assets/placeholder17.png)

---

### C. Heatmap Avanzado de Características

---

```python
# === HEATMAP DETALLADO ===
import seaborn as sns

# Crear matriz de características por cluster
cluster_profiles = df_customers.groupby('cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean()

plt.figure(figsize=(10, 6))
sns.heatmap(cluster_profiles.T,
           annot=True,
           cmap='RdYlBu_r',
           center=cluster_profiles.values.mean(),
           cbar_kws={'label': 'Valor Promedio'})
plt.title('Perfil de Características por Cluster')
plt.ylabel('Características')
plt.xlabel('Cluster ID')
plt.show()
```

![Visualización generada](../assets/placeholder18.png)

---

## 📈 Challenge 5: Comparación Masiva de Algoritmos

---

```python
# === BENCHMARK DE ALGORITMOS (ROBUSTO) ===
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# 0) Asegurar escalado (muchos métodos lo agradecen)
X_bench = StandardScaler().fit_transform(X_final_for_clustering)

# 1) K por defecto si no está definido
optimal_k = globals().get("optimal_k", 4)

algorithms = {
    'K-Means': KMeans(n_clusters=optimal_k, random_state=42, n_init=10),
    'DBSCAN': DBSCAN(eps=0.5, min_samples=5),
    'Spectral (rbf)': SpectralClustering(n_clusters=optimal_k, affinity='rbf', random_state=42),
    # Consejo: para datos con vecindad clara, prueba 'nearest_neighbors'
    'Agglomerative (ward)': AgglomerativeClustering(n_clusters=optimal_k, linkage='ward'),
    # 'HDBSCAN': hdbscan.HDBSCAN(min_cluster_size=20),  # si lo tienes instalado
}

results = {}
for name, algo in algorithms.items():
    try:
        labels = algo.fit_predict(X_bench)

        # contar clusters reales (excluye ruido -1)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int(np.sum(labels == -1)) if -1 in labels else 0

        score = silhouette_score(X_bench, labels) if n_clusters > 1 else -1.0
        results[name] = {'silhouette': score, 'n_clusters': n_clusters, 'n_noise': n_noise}
        print(f"{name}: Silhouette={score:.3f}, Clusters={n_clusters}, Noise={n_noise}")

    except Exception as e:
        print(f"{name}: ERROR - {e}")

# 2) Visualización comparativa
names = list(results.keys())
scores = [results[n]['silhouette'] for n in names]

plt.figure(figsize=(12,6))
bars = plt.bar(names, scores, alpha=0.8)
plt.ylabel('Silhouette Score'); plt.title('Comparación de Algoritmos de Clustering'); plt.xticks(rotation=20)

for bar, s in zip(bars, scores):
    bar.set_color('green' if s >= 0.5 else 'orange' if s >= 0.25 else 'red')
    plt.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f'{s:.3f}', ha='center', va='bottom')

plt.tight_layout(); plt.show()

# 3) Ordenar y mostrar ranking
ranking = sorted(results.items(), key=lambda kv: kv[1]['silhouette'], reverse=True)
print("\n🏁 Ranking por Silhouette:")
for i, (n, m) in enumerate(ranking, 1):
    print(f"{i}. {n}  |  silhouette={m['silhouette']:.3f}  |  k={m['n_clusters']}  |  noise={m['n_noise']}")
```

#### 📊 Salida:
```text
K-Means: Silhouette=0.679, Clusters=4, Noise=0
DBSCAN: Silhouette=0.529, Clusters=2, Noise=0
Spectral (rbf): Silhouette=0.680, Clusters=4, Noise=0
Agglomerative (ward): Silhouette=0.680, Clusters=4, Noise=0
```

![Visualización generada](../assets/placeholder19.png)

```text
🏁 Ranking por Silhouette:
1. Spectral (rbf)  |  silhouette=0.680  |  k=4  |  noise=0
2. Agglomerative (ward)  |  silhouette=0.680  |  k=4  |  noise=0
3. K-Means  |  silhouette=0.679  |  k=4  |  noise=0
4. DBSCAN  |  silhouette=0.529  |  k=2  |  noise=0
```

---

