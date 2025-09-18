---
title: "Práctica 2: Feature Engineering simple + Modelo base"
date: 2025-01-01
---

# ⚙️ Práctica 2 · Feature Engineering simple + Modelo base

<div class="grid cards" markdown>

-   :material-database-cog: **Dataset**
    ---
    Titanic (Kaggle) con variables demográficas y socioeconómicas.

-   :material-notebook: **Notebook**
    ---
    [Abrir en Colab](https://colab.research.google.com/drive/1ut5NvjzklgNwS8wfOD07xslXUY7flhu4?usp=sharing#scrollTo=feature-engineering)

-   :material-account-badge: **Rol**
    ---
    Diseño de features y comparación de modelos base (DummyClassifier vs LogisticRegression).

-   :material-flag-checkered: **Estado**
    ---
    ✅ Entregado

</div>

## En una mirada

- Se consolidó un pipeline reproducible que prepara datos, crea nuevas variables y entrena modelos comparables.
- La Regresión Logística alcanzó un accuracy del **81.5 %**, superando ampliamente al baseline de clase mayoritaria (61 %).
- La matriz de confusión permitió priorizar mejoras enfocadas en reducir falsos negativos.

!!! tip "Métrica a seguir"
    En futuras iteraciones, optimizar **recall** para disminuir falsos negativos en la clase positiva (sobrevivientes).

## 🎯 Objetivos

- Practicar la creación de nuevas variables (*features*) a partir de los datos originales.
- Construir un modelo base de clasificación con **Logistic Regression** y compararlo con un modelo trivial.
- Evaluar métricas de rendimiento más allá de la accuracy (precision, recall, F1, matriz de confusión).

## 🗓️ Agenda express

| Actividad | Propósito | Tiempo |
|-----------|-----------|:------:|
| Revisión del dataset y limpieza | Analizar valores faltantes y preparar columnas. | 20 min |
| Creación de nuevas features | Generar `FamilySize`, `IsAlone`, `Title` y codificar categorías. | 30 min |
| Entrenamiento del modelo base | Ajustar LogisticRegression con hiperparámetros controlados. | 25 min |
| Evaluación de métricas | Comparar resultados con DummyClassifier y extraer aprendizajes. | 20 min |

## 🔍 Insights destacados

- El baseline con `DummyClassifier` fija el piso en **61 %** de accuracy.
- Las variables creadas (`FamilySize`, `IsAlone`, `Title`) aportan señal y elevan el desempeño al 81.5 %.
- La matriz de confusión revela **21 falsos negativos** → foco para siguientes iteraciones.

## 🛠️ Desarrollo guiado

### 1. Contexto del algoritmo

#### LogisticRegression

- **¿Qué problemas resuelve?** Clasificación binaria y multiclase.
- **Parámetros relevantes:** `penalty`, `solver`, `max_iter`.
- **¿Cuándo usar `solver='liblinear'`?** Adecuado para datasets pequeños/medianos y para penalización L1; otros solvers (`lbfgs`, `newton-cg`) sólo soportan L2 o ninguna.

#### train_test_split

- `stratify` preserva la proporción de clases en train/test.
- `random_state` asegura reproducibilidad.
- Proporción recomendada: **70/30** u **80/20** según tamaño de muestra.

### 2. Configuración de entorno y datos

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

```text
Mounted at /content/drive
Outputs → /content/drive/MyDrive/IA-UT1
```

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

```text
Se eligió un archivo
Upload widget is only available when the cell has been executed in the current browser session. Please rerun this cell to enable.
Saving kaggle.json to kaggle.json
Downloading titanic.zip to data
  0% 0.00/34.1k [00:00<?, ?B/s]
100% 34.1k/34.1k [00:00<00:00, 165MB/s]
Archive:  data/titanic.zip
  inflating: data/gender_submission.csv
  inflating: data/test.csv
  inflating: data/train.csv
```

### 3. Feature engineering paso a paso

```python
df = train.copy()

# PASO 1 · Imputación
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
df['Age'] = df['Age'].fillna(df.groupby(['Sex', 'Pclass'])['Age'].transform('median'))

# PASO 2 · Nuevas variables
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

df['Title'] = df['Name'].str.extract(',\s*([^\.]+)\.')
rare_titles = df['Title'].value_counts()[df['Title'].value_counts() < 10].index
df['Title'] = df['Title'].replace(rare_titles, 'Rare')

# PASO 3 · Preparación para el modelo
features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'FamilySize', 'IsAlone', 'Title', 'SibSp', 'Parch']
X = pd.get_dummies(df[features], drop_first=True)
y = df['Survived']

X.shape, y.shape
```

```text
((891, 14), (891,))
```

!!! note "Interpretación"
    - Se mantienen las 891 observaciones originales.
    - Tras la codificación, se generan **14 variables predictoras** listas para modelar.

### 4. Entrenamiento y evaluación

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

dummy = DummyClassifier(strategy='most_frequent', random_state=42)
dummy.fit(X_train, y_train)

log_reg = LogisticRegression(max_iter=1000, solver='liblinear')
log_reg.fit(X_train, y_train)

baseline_acc = accuracy_score(y_test, dummy.predict(X_test))
model_acc = accuracy_score(y_test, log_reg.predict(X_test))
```

```python
print(f"Baseline accuracy: {baseline_acc:.3f}")
print(f"Logistic Regression accuracy: {model_acc:.3f}")
print(classification_report(y_test, log_reg.predict(X_test)))
print(confusion_matrix(y_test, log_reg.predict(X_test)))
```

!!! success "Resultados"
    - **Baseline:** 0.611
    - **Logistic Regression:** 0.815
    - 21 falsos negativos identificados en la matriz de confusión.

## ✅ Cierre y próximos pasos

- Documentación del pipeline lista para iterar con validación cruzada.
- Prioridad: explorar regularización y ajuste de umbral para mejorar recall.
- Integrar visualizaciones de importancia de features para presentar a stakeholders.
