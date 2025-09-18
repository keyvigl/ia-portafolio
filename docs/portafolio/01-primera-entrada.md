---
title: "Práctica 1: EDA del Titanic en Google Colab"
date: 2025-01-01
---

# 🧊 Práctica 1 · EDA del Titanic en Google Colab

<div class="grid cards" markdown>

-   :material-database: **Dataset**
    ---
    Titanic – Machine Learning from Disaster (Kaggle).

-   :material-notebook: **Notebook**
    ---
    [Abrir en Colab](https://colab.research.google.com/drive/1ut5NvjzklgNwS8wfOD07xslXUY7flhu4?usp=sharing)

-   :material-account-badge: **Rol**
    ---
    Análisis exploratorio con foco en limpieza, visualización e insights accionables.

-   :material-flag-checkered: **Estado**
    ---
    ✅ Entregado

</div>

## En una mirada

- Se automatizó la descarga del dataset mediante Kaggle API y se organizó la estructura de carpetas en Drive.
- El EDA permitió identificar distribución de variables, valores faltantes críticos y primeras hipótesis de supervivencia.
- Se documentaron hallazgos en tablas interpretables para facilitar iteraciones futuras.

!!! success "Insight clave"
    La combinación **sexo + clase** es el indicador más fuerte de supervivencia; se usará como punto de partida para futuros modelos.

## 🎯 Objetivos

- Conocer la estructura del dataset y sus principales variables.
- Practicar **EDA (Exploratory Data Analysis)** con Pandas, Matplotlib y Seaborn.
- Identificar factores relevantes para la supervivencia y documentarlos claramente.
- Detectar problemas de calidad de datos (faltantes, outliers, correlaciones) de cara a futuros modelos.

## 🗓️ Agenda express

| Actividad | Propósito | Tiempo |
|-----------|-----------|:------:|
| Investigación del dataset | Revisar documentación y entender el reto de Kaggle. | 10 min |
| Setup en Colab | Preparar dependencias y estilo de visualización. | 5 min |
| Descarga y carga de datos | Automatizar acceso al dataset vía Kaggle API. | 10 min |
| EDA descriptiva y visual | Explorar variables clave, distribuciones y correlaciones. | 15 min |
| Documentación y discusión | Registrar hallazgos y preparar conclusiones. | 10 min |

## 🧱 Contexto rápido

Exploramos el dataset **Titanic: Machine Learning from Disaster** de Kaggle. Se trata de un problema de **clasificación binaria** que busca predecir si un pasajero sobrevivió (`Survived = 1`) o no (`Survived = 0`). Es un clásico introductorio porque combina variables demográficas, socioeconómicas y familiares.

## 🔍 Insights destacados

- Mayor probabilidad de supervivencia en mujeres, niños y pasajeros de primera clase.
- Columnas `Age`, `Cabin` y `Embarked` requieren estrategias de imputación específicas antes del modelado.
- Variables socioeconómicas (`Pclass`, `Fare`) muestran correlaciones útiles para construir características derivadas.

## 🛠️ Desarrollo paso a paso

### 1. Preparación del entorno

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette('deep')
```

### 2. Configuración de rutas y Google Drive

```python
from pathlib import Path
try:
    from google.colab import drive
    drive.mount('/content/drive')
    ROOT = Path('/content/drive/MyDrive/IA-UT1')
except Exception:
    ROOT = Path.cwd() / 'IA-UT1'

DATA_DIR = ROOT / 'data'
RESULTS_DIR = ROOT / 'results'
for d in (DATA_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

print('Outputs →', ROOT)
```

#### Salida

```text
Mounted at /content/drive
Outputs → /content/drive/MyDrive/IA-UT1
```

### 3. Descarga del dataset desde Kaggle

```python
!pip -q install kaggle
from google.colab import files
files.upload()  # Subí tu archivo kaggle.json descargado

!mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
!kaggle competitions download -c titanic -p data
!unzip -o data/titanic.zip -d data

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
```

#### 💻 Salida

```text
se eligió ningún archivo
Upload widget is only available when the cell has been executed in the current browser session. Please rerun this cell to enable.
Saving kaggle.json to kaggle.json
Downloading titanic.zip to data
  0% 0.00/34.1k [00:00<?, ?B/s]
100% 34.1k/34.1k [00:00<00:00, 124MB/s]
Archive:  data/titanic.zip
  inflating: data/gender_submission.csv
  inflating: data/test.csv
  inflating: data/train.csv
```

### 4. Conociendo el dataset

```python
train.shape, train.columns
```

```text
(891, 12)
Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
      dtype='object')
```

!!! note "Interpretación"
    - **Tamaño:** 891 filas × 12 columnas.
    - **Tipos:** mezcla de variables numéricas y categóricas.
    - **Objetivo:** `Survived` (0 = no, 1 = sí).

```python
train.head(3)
```

```text
   PassengerId  Survived  Pclass  Name                                         Sex     Age  SibSp  Parch     Ticket         Fare  Cabin Embarked
0            1         0       3  Braund, Mr. Owen Harris                male    22.0      1      0  A/5 21171      7.2500    NaN       S
1            2         1       1  Cumings, Mrs. John Bradley ...         female  38.0      1      0  PC 17599      71.2833    C85       C
2            3         1       3  Heikkinen, Miss. Laina                female  26.0      0      0  STON/O2. 3101282  7.9250    NaN       S
```

!!! abstract "Qué observamos"
    - Cada fila representa un pasajero con atributos demográficos y socioeconómicos.
    - Los ejemplos iniciales muestran supervivencias distintas, reforzando la importancia de `Sex`, `Pclass` y `Age`.

```python
train.info()
```

```text
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
 #   Column       Non-Null Count  Dtype
---  ------       --------------  -----
 0   PassengerId  891 non-null    int64
 1   Survived     891 non-null    int64
 2   Pclass       891 non-null    int64
 3   Name         891 non-null    object
 4   Sex          891 non-null    object
 5   Age          714 non-null    float64
 6   SibSp        891 non-null    int64
 7   Parch        891 non-null    int64
 8   Ticket       891 non-null    object
 9   Fare         891 non-null    float64
 10  Cabin        204 non-null    object
 11  Embarked     889 non-null    object
dtypes: float64(2), int64(5), object(5)
memory usage: 83.7+ KB
```

!!! warning "Valores faltantes críticos"
    - `Age`: 177 registros sin dato.
    - `Cabin`: más del 75 % vacío → requiere estrategia específica.
    - `Embarked`: 2 registros faltantes, fáciles de imputar.

## 🧭 Preguntas guía

| Pregunta | Respuesta | Observación |
|----------|-----------|-------------|
| **¿Qué es el dataset del Titanic?** | Conjunto de datos histórico para practicar **clasificación supervisada**. | Ideal como primer proyecto: datos reales, estructura simple. |
| **¿De qué trata exactamente?** | Usa datos de pasajeros (edad, sexo, clase, familiares, etc.) para predecir supervivencia. | Mezcla variables demográficas y socioeconómicas. |
| **¿Cuál es el objetivo de la competencia?** | Introducir a participantes en ciencia de datos y machine learning. | Funciona como reto introductorio y benchmark clásico. |

## 💡 Hallazgos prioritarios

1. **Factores influyentes**
   - 🚹🚺 Sexo: las mujeres tuvieron mayor supervivencia.
   - 🎟️ Clase: primera clase con ventaja notable.
   - 👶 Edad: prioridad para niños.
2. **Desafíos de calidad**
   - ❌ Faltantes en `Age`, `Cabin`, `Embarked`.
   - 📊 Formatos por estandarizar (tickets, cabinas).
   - 🕰️ Posible sesgo histórico.
3. **Correlaciones útiles**
   - 💰 `Pclass` y `Fare` como indicadores socioeconómicos.
   - 🔗 Interacción sexo × clase.

## ✅ Cierre y próximos pasos

- Documentación lista para iterar sobre imputación y modelos.
- Prioridad: definir estrategia de valores faltantes antes de entrenar.
- Seguir con feature engineering aprovechando insights encontrados.
