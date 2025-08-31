
---
title: "Práctica 2: Feature Engineering simple + Modelo base"
date: 2025-01-01
---

# ⚙️ Práctica 2: Feature Engineering simple + Modelo base

## Contexto
En esta práctica trabajamos con el dataset del Titanic para aplicar un **proceso de Feature Engineering simple** y entrenar un **modelo base de clasificación**.  
Se busca entender cómo transformar variables, crear nuevas características y evaluar un modelo inicial.

## Objetivos
- Practicar la creación de nuevas variables (*features*) a partir de los datos originales.  
- Construir un modelo base de clasificación (Logistic Regression).  
- Comparar su desempeño con un modelo trivial (*DummyClassifier*).  
- Evaluar métricas de rendimiento más allá de la accuracy.  
## Actividades (con tiempos estimados)
| Actividad                     | Tiempo |
|--------------------------------|:------:|
| Revisión del dataset y limpieza | 20 min |
| Creación de nuevas features     | 30 min |
| Entrenamiento del modelo base   | 25 min |
| Evaluación de métricas          | 20 min |


## Desarrollo
## 🔎 LogisticRegression

### ❓ ¿Qué tipo de problema resuelve?
Los problemas de **clasificación**, de manera más específica la **clasificación binaria**.

---

### ⚙️ ¿Qué parámetros importantes tiene?
- **penalty** → tipo de regularización.  
- **solver** → algoritmo de optimización para estimar coeficientes.  
- **max_iter** → número máximo de iteraciones del solver.  

---

### 🗝️ ¿Cuándo usar `solver='liblinear'` vs otros solvers?
- Principalmente se puede usar para **problemas pequeños o medianos**.  
- Permite usar penalización **L1**, dando modelos más dispersos (más coeficientes en cero).  
- Otros solvers como **lbfgs** o **newton-cg** solo soportan penalización L2 o incluso ninguna.  


## 🔀 train_test_split

### ❓ ¿Qué hace el parámetro `stratify`?
Asegura que la proporción de clases en los conjuntos de entrenamiento y prueba sea la misma que en el dataset completo.

---

### 🎲 ¿Por qué usar `random_state`?
- Poder obtener los mismos resultados y compartirlos o depurar con consistencia.  
- Evaluar diferentes modelos o parámetros sobre la misma división de datos.  

---

### 📊 ¿Qué porcentaje de test es recomendable?
La proporción más comúnmente recomendada es usar entre **20 % y 30 %** del total para testing  
(lo que implica usar entre **70 % y 80 %** para entrenamiento).  

## 📏 Métricas de evaluación

### ❓ ¿Qué significa cada métrica en `classification_report`?
- **Precision**: mide la proporción de predicciones positivas que fueron correctas.  
- **Recall**: indica qué fracción de instancias positivas reales fueron correctamente identificadas.  
- **F1-score**: la media entre precision y recall, ofreciendo un balance entre ambos.  

---

### 🔎 ¿Cómo interpretar la matriz de confusión?
- **Filas** → clases reales.  
- **Columnas** → clases predichas.  

---

### ⚖️ ¿Cuándo usar accuracy vs otras métricas?
- **Accuracy** es intuitiva pero a veces engañosa.  
- Cuando las clases están **desbalanceadas**, es mejor usar **precisión, recall, F1** o métricas más sofisticadas que consideren costos distintos o reemplacen el desequilibrio.  


### 📂 Configuración de rutas en Google Colab

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
### 📥 Descarga del dataset Titanic con Kaggle API

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
#### Salida:
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
### 🛠️ Feature Engineering y preparación de datos

```python
df = train.copy()

# 🚫 PASO 1: Manejar valores faltantes (imputación)
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])  # Valor más común
df['Fare'] = df['Fare'].fillna(df['Fare'].median())              # Mediana
df['Age'] = df['Age'].fillna(df.groupby(['Sex','Pclass'])['Age'].transform('median'))

# 🆕 PASO 2: Crear nuevas features útiles
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

df['Title'] = df['Name'].str.extract(',\s*([^\.]+)\.')
rare_titles = df['Title'].value_counts()[df['Title'].value_counts() < 10].index
df['Title'] = df['Title'].replace(rare_titles, 'Rare')

# 🔄 PASO 3: Preparar datos para el modelo
features = ['Pclass','Sex','Age','Fare','Embarked','FamilySize','IsAlone','Title','SibSp','Parch']
X = pd.get_dummies(df[features], drop_first=True)
y = df['Survived']

X.shape, y.shape
```
#### Salida:
```text
((891, 14), (891,))
```
#### 📌 Interpretación:
- El dataset final tiene **891 observaciones** (pasajeros).  
- Después del *feature engineering* y la codificación con *dummies*, se generaron **14 columnas predictoras** listas para usar en el modelo.  
- La variable objetivo **`y`** contiene las etiquetas de supervivencia para los mismos 891 pasajeros.  
- ✅ Esto confirma que no se perdieron filas tras la imputación y transformación de variables.  

### 🤖 Entrenamiento y evaluación de modelos

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
baseline_pred = dummy.predict(X_test)

lr = LogisticRegression(max_iter=1000, solver='liblinear', random_state=42)
lr.fit(X_train, y_train)
pred = lr.predict(X_test)

print('Baseline acc:', accuracy_score(y_test, baseline_pred))
print('LogReg acc  :', accuracy_score(y_test, pred))

print('\nClassification report (LogReg):')
print(classification_report(y_test, pred))

print('\nConfusion matrix (LogReg):')
print(confusion_matrix(y_test, pred))
```

#### Salida:
```text
Baseline acc: 0.6145251396648045
LogReg acc  : 0.8156424581005587

Classification report (LogReg):
              precision    recall  f1-score   support

           0       0.82      0.89      0.86       110
           1       0.80      0.70      0.74        69

    accuracy                           0.82       179
   macro avg       0.81      0.79      0.80       179
weighted avg       0.81      0.82      0.81       179


Confusion matrix (LogReg):
[[98 12]
 [21 48]]

```
#### 📌 Interpretación:
- El modelo base (DummyClassifier) obtiene un accuracy de aproximadamente 61 %, equivalente a predecir siempre la clase mayoritaria (no sobrevivió).  
- La Regresión Logística mejora el rendimiento con un accuracy de aproximadamente 81.5 %, lo que representa una ganancia importante frente al baseline.  
- Según la matriz de confusión:  
  - Se acierta en la mayoría de los casos de pasajeros que no sobrevivieron (98).  
  - Se identifican correctamente 48 casos de pasajeros que sí sobrevivieron.  
  - Se producen 21 falsos negativos (personas que sobrevivieron pero fueron clasificadas como no sobrevivientes).  
  - Se producen 12 falsos positivos (personas que no sobrevivieron pero fueron clasificadas como sobrevivientes).  
- En las métricas, la clase 0 (no sobrevivió) alcanza un recall alto (0.89), mientras que la clase 1 (sí sobrevivió) tiene un recall menor (0.70), lo que indica que el modelo falla más en identificar sobrevivientes.  
- En conclusión, la Regresión Logística supera ampliamente al modelo base, aunque aún presenta limitaciones en la detección de casos positivos de supervivencia.  



### ❓ Preguntas para el equipo

**Matriz de confusión: ¿En qué casos se equivoca más el modelo: cuando predice que una persona sobrevivió y no lo hizo, o al revés?**  
El modelo se equivoca más al predecir que alguien no sobrevivió cuando en realidad sí sobrevivió (21 casos) que al revés (12 casos).

---

**Clases atendidas: ¿El modelo acierta más con los que sobrevivieron o con los que no sobrevivieron?**  
- No sobrevivieron (0) → 98/110 ≈ 89 % de aciertos  
- Sobrevivieron (1) → 48/69 ≈ 70 % de aciertos  
El modelo acierta más con las personas que no sobrevivieron.

---

**Comparación con baseline: ¿La Regresión Logística obtiene más aciertos que el modelo que siempre predice la clase más común?**  
La Regresión Logística mejora significativamente sobre el baseline, confirmando que el modelo está aprendiendo patrones útiles.

---

**Errores más importantes: ¿Cuál de los dos tipos de error creés que es más grave para este problema?**  
El error más crítico son los **falsos negativos (FN)** → predecir que alguien no sobrevivirá cuando sí lo hizo.  
En un escenario real (simulación de rescate), este error sería más grave porque significa no salvar a alguien que podría sobrevivir.

---

**Observaciones generales: Mirando las gráficas y números, ¿qué patrones interesantes encontraste sobre la supervivencia?**  
- Sexo: las mujeres tenían mucha más probabilidad de sobrevivir que los hombres.  
- Clase: los pasajeros de primera clase tuvieron mayor supervivencia, los de tercera clase menor.  
- Edad: los niños tenían más probabilidad de sobrevivir.

---

**Mejoras simples: ¿Qué nueva columna (feature) se te ocurre que podría ayudar a que el modelo acierte más?**  
- Interacción entre `Sex` y `Pclass`: algunas clases de mujeres tenían más prioridad de rescate.  
- Binning de edad: categorizar en niños, jóvenes, adultos y ancianos para capturar patrones no lineales.  
- Presencia de `Cabin`: si la persona tenía número de camarote, podría indicar clase alta y, por lo tanto, mayor supervivencia.  


## 📂 Evidencias
- Capturas de pantallas del proceso y gráficos guardados en `docs/assets/`.  
- Notebook en Google Colab con el desarrollo completo de la práctica:  
[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ut5NvjzklgNwS8wfOD07xslXUY7flhu4?usp=sharing)  


---

## 🤔 Reflexión
- **Lo más desafiante:** manejar los valores faltantes y diseñar nuevas variables útiles que aportaran al modelo.  
- **Lo más valioso:** comprobar cómo el *feature engineering* mejora significativamente el rendimiento respecto al modelo base.  
- **Próximos pasos:** probar interacciones entre variables (ej. `Sex × Pclass`), aplicar *binning* a la edad y evaluar modelos más complejos.  

