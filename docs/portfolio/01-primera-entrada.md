---
title: "Práctica 1: EDA del Titanic en Google Colab"
date: 2025-01-01
---

# Práctica 1: EDA del Titanic en Google Colab

## Contexto
Exploramos el dataset **Titanic: Machine Learning from Disaster** de Kaggle.  
Se trata de un problema de **clasificación binaria**: predecir si un pasajero sobrevivió (`Survived=1`) o no (`Survived=0`).  
La competencia busca que los participantes apliquen técnicas de EDA, ingeniería de variables y modelos de machine learning.

## Objetivos
- Conocer la estructura del dataset.
- Practicar **EDA (Exploratory Data Analysis)** con Pandas, Matplotlib y Seaborn.
- Identificar factores relevantes para la supervivencia.
- Detectar problemas de calidad de datos (faltantes, outliers, correlaciones).

## Actividades (con tiempos estimados)
- Investigación del dataset — 10 min  
- Setup en Colab — 5 min  
- Cargar datos con Kaggle API — 10 min  
- EDA descriptiva y visual — 15 min  
- Documentación y discusión — 10 min  

## Desarrollo
### 📖 Investigación inicial

| Pregunta | Respuesta | Observación |
|----------|-----------|-------------|
| **¿Qué es el dataset del Titanic?** | Es un conjunto de datos histórico muy usado en **ciencia de datos y machine learning** para practicar problemas de clasificación supervisada. | Ideal como primer proyecto: datos reales, pero con estructura simple. |
| **¿De qué trata exactamente este dataset?** | Se basa en información de los pasajeros (edad, sexo, clase, tarifa, familiares a bordo, etc.) para predecir si habrían sobrevivido o no al hundimiento del Titanic. | Combina variables demográficas y socioeconómicas. |
| **¿Cuál es el objetivo de la competencia de Kaggle?** | Que cualquier persona —incluso sin mucha experiencia— pueda iniciarse en **ciencia de datos y aprendizaje automático**, aplicando desde EDA hasta modelos de clasificación. | Funciona como un reto introductorio y un benchmark clásico. |

### 🤔 Preguntas de investigación

**1. Factores que más influyeron en la supervivencia**
- 🚹🚺 **Sexo:** las mujeres tuvieron mucha mayor probabilidad de sobrevivir.  
- 🎟️ **Clase de boleto:** primera clase con mayor supervivencia.  
- 👶 **Edad:** los niños recibieron prioridad en el rescate.  

**2. Desafíos de calidad de datos**
- ❌ Valores faltantes (especialmente en `Age`, `Cabin`, `Embarked`).  
- 📊 Formatos inconsistentes que requieren limpieza.  
- 🕰️ Dataset histórico con posible sesgo en los registros.  

**3. Posibles correlaciones**
- 💰 `Pclass` y `Fare` como indicadores socioeconómicos.  
- 📉 Mayor supervivencia en primera clase comparado con tercera.  
- 🔗 Interacción entre **sexo y clase social**.

### ⚙️ 1. Setup inicial

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
### 📂 2. Configuración de rutas y Google Drive

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
#### Salida:
```text
Mounted at /content/drive
Outputs → /content/drive/MyDrive/IA-UT1
```
### 📥 3. Cargar el dataset de Kaggle

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
#### 💻 Salida:
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
### 🔎 3. Conocer el dataset

```python
train.shape, train.columns
```

```text
(891, 12)
Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
      dtype='object')
```
📌 Interpretación:
- El dataset de entrenamiento contiene 891 filas (cada fila = un pasajero).
- Tiene 12 columnas, que incluyen variables de identificación (PassengerId), demográficas (Sex, Age), socioeconómicas (Pclass, Fare, Cabin, Embarked) y familiares (SibSp, Parch).
- La variable objetivo es Survived (0 = no sobrevivió, 1 = sí sobrevivió).
```python
train.head(3)
```
#### 💻 Salida:
```text
   PassengerId  Survived  Pclass  Name                                         Sex     Age  SibSp  Parch     Ticket         Fare Cabin Embarked
0            1         0       3  Braund, Mr. Owen Harris                male    22.0      1      0  A/5 21171      7.2500   NaN       S
1            2         1       1  Cumings, Mrs. John Bradley ...         female  38.0      1      0  PC 17599        71.2833  C85       C
2            3         1       3  Heikkinen, Miss. Laina                female  26.0      0      0  STON/O2. 3101282  7.9250  NaN       S
```
 📌 Interpretación:
- Cada fila representa un pasajero.  
- Las columnas incluyen datos demográficos, socioeconómicos y familiares.  
- En esta muestra:  
  - El pasajero 1 (hombre, 22 años, 3ª clase) **no sobrevivió**.  
  - El pasajero 2 (mujer, 38 años, 1ª clase, cabina C85) **sí sobrevivió**.  
  - El pasajero 3 (mujer, 26 años, 3ª clase) **sí sobrevivió**.  
```python
train.info()
```
#### 💻 Salida:
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
📌 Interpretación:
- 📏 **Tamaño**: 891 filas (pasajeros) × 12 columnas (atributos).  
- 🔢 **Tipos de datos**:
  - `int64`: 5 columnas (ej. `Survived`, `Pclass`).  
  - `float64`: 2 columnas (`Age`, `Fare`).  
  - `object`: 5 columnas (`Name`, `Sex`, `Ticket`, etc.).  
- ⚠️ **Valores faltantes**:
  - `Age`: 714 no nulos → faltan 177 valores.  
  - `Cabin`: solo 204 no nulos → más del 75 % de datos perdidos.  
  - `Embarked`: 889 no nulos → faltan 2 valores.  
- 💾 **Uso de memoria**: 83.7 KB → dataset ligero y manejable en Colab.

> ✅ Con esta exploración identificamos que `Age` y `Embarked` necesitan **imputación**, y `Cabin` puede requerir un tratamiento especial (ej. variable binaria *Cabin known / unknown*).
```python
train.describe(include='all').T
```
```text
              count unique                     top freq       mean        std   min      25%   50%      75%      max
PassengerId   891.0    NaN                     NaN  NaN  446.000000  257.353842   1.0   223.5  446.0   668.5   891.0
Survived      891.0    NaN                     NaN  NaN    0.383838    0.486592   0.0     0.0    0.0     1.0     1.0
Pclass        891.0    NaN                     NaN  NaN    2.308642    0.836071   1.0     2.0    3.0     3.0     3.0
Name            891    891    Dooley, Mr. Patrick    1        NaN         NaN     NaN     NaN    NaN     NaN     NaN
Sex            891      2                   male  577        NaN         NaN     NaN     NaN    NaN     NaN     NaN
Age           714.0    NaN                     NaN  NaN   29.699118   14.526497   0.42   20.125  28.0   38.0    80.0
SibSp         891.0    NaN                     NaN  NaN    0.523008    1.102743   0.0     0.0    0.0     1.0     8.0
Parch         891.0    NaN                     NaN  NaN    0.381594    0.806057   0.0     0.0    0.0     0.0     6.0
Ticket         891    681                  347082    7        NaN         NaN     NaN     NaN    NaN     NaN     NaN
Fare          891.0    NaN                     NaN  NaN   32.204208   49.693429   0.0     7.9104 14.4542  31.0  512.3292
Cabin          204    147                      G6    4        NaN         NaN     NaN     NaN    NaN     NaN     NaN
Embarked       889      3                       S  644        NaN         NaN     NaN     NaN    NaN     NaN     NaN

```
 📌 Interpretación:
- 📏 **Distribución de pasajeros**: 891 en total.  
- 🎯 **Survived**: la media ≈ 0.38 → alrededor del 38 % sobrevivieron y 62 % no.  
- 🎟️ **Pclass**: promedio ≈ 2.3 → mayoría viajaba en 3ª clase.  
- 🚹🚺 **Sexo**: 577 hombres y 314 mujeres.  
- 👶 **Edad**: promedio ≈ 29.7 años, rango entre 0.42 y 80 años; faltan 177 registros.  
- 👨‍👩‍👧 **Familiares a bordo** (`SibSp`, `Parch`): la mayoría viajó sola (valores = 0).  
- 💵 **Fare**: media ≈ 32, gran dispersión (mínimo 0, máximo > 500).  
- 🛏️ **Cabin**: solo 204 registros completos → más del 75 % faltante.  
- ⛴️ **Embarked**: 3 categorías; predominante `S` (Southampton) con 644 pasajeros.

> ✅ Este análisis revela patrones socioeconómicos (sexo, clase, tarifa) y problemas de calidad de datos (faltantes en `Age`, `Cabin` y `Embarked`) que deben abordarse en el preprocesamiento.

```python
train.isna().sum().sort_values(ascending=False)
```

```text
Cabin          687
Age            177
Embarked         2
PassengerId      0
Name             0
Pclass           0
Survived         0
Sex              0
Parch            0
SibSp            0
Fare             0
Ticket           0
dtype: int64

```

 📌 Interpretación:
- 🛏️ **Cabin**: 687 valores faltantes → más del **75 %** → difícil de usar directamente.  
- 👶 **Age**: 177 valores faltantes → requiere **imputación** (ej. mediana por grupo de clase/sexo).  
- ⛴️ **Embarked**: 2 valores faltantes → se puede completar con la moda.  
- ✅ El resto de variables **no tienen valores nulos**.  

> ✅ Este análisis confirma que `Cabin` podría transformarse en una variable binaria (*tiene/no tiene registro*), mientras que `Age` y `Embarked` deben ser imputadas para evitar pérdida de información.

```python
train['Survived'].value_counts(normalize=True)
```

```text
Survived
0    0.616162
1    0.383838
Name: proportion, dtype: float64
```

 📌 Interpretación:
- ⚰️ **No sobrevivieron (0): ~61.6 %** de los pasajeros.  
- 🛟 **Sobrevivieron (1): ~38.4 %** de los pasajeros.  
- 📉 Existe un **desbalance moderado**: si un modelo predijera siempre "no sobrevivió", lograría ≈61 % de accuracy.  
- ✅ Esto obliga a usar métricas adicionales (precision, recall, F1) en lugar de solo accuracy.

### 📊 4. Análisis exploratorio visual (EDA)

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Supervivencia global por sexo
sns.countplot(data=train, x='Survived', hue='Sex', ax=axes[0,0])
axes[0,0].set_title('Supervivencia por sexo')

# Tasa de supervivencia por clase
sns.barplot(data=train, x='Pclass', y='Survived', estimator=np.mean, ax=axes[0,1])
axes[0,1].set_title('Tasa de supervivencia por clase')

# Distribución de edad por supervivencia
sns.histplot(data=train, x='Age', hue='Survived', kde=True, bins=30, ax=axes[1,0])
axes[1,0].set_title('Edad vs supervivencia')

# Correlaciones numéricas
numeric_cols = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
sns.heatmap(train[numeric_cols].corr(), annot=True, cmap='Blues', ax=axes[1,1])
axes[1,1].set_title('Correlaciones')

plt.tight_layout()
plt.show()
```
### Salida

![Evidencia](assets/01EDA.png)


### ❓ Preguntas para el equipo

**1. ¿Qué variables parecen más relacionadas con `Survived`?**  
- 🚹🚺 **Sexo**: la variable más fuerte → las mujeres tuvieron mayor tasa de supervivencia.  
- 🎟️ **Pclass / Fare**: reflejan posición socioeconómica.  
- 👶 **Edad**: los niños y jóvenes tuvieron más chances de sobrevivir.  

**2. ¿Dónde hay más valores faltantes? ¿Cómo los imputarías?**  
- 🛏️ **Cabin**: muchísimos valores faltantes → mejor crear variable binaria `HasCabin`.  
- 👶 **Age**: valores faltantes moderados → imputar con la mediana segmentada (por `Title` o `Pclass`).  

**3. ¿Qué hipótesis probarías a continuación?**  
- 🎭 **Extraer `Title`** de los nombres puede mejorar imputación de edad y performance del modelo.  
- 🔗 **Interacción `Sex × Pclass`**: mujeres de 3ª clase tuvieron menor supervivencia que mujeres de 1ª clase.





## Evidencias
- Enlace a material o capturas en `docs/assets/`

## Reflexión
Lo más desafiante, lo más valioso, próximos pasos.
