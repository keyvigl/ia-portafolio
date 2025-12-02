---
title: "🧪 Práctica 14 — Introducción a LangChain con OpenAI: Prompting, Plantillas y RAG"
date: 2025-10-28
---

# 🧪 Práctica 14 — Introducción a LangChain con OpenAI: Prompting, Plantillas y RAG  
**Unidad Temática 4 — LLMs, LangChain y OpenAI**

---

## 📘 Contexto General

Esta práctica introduce el uso de **LangChain** integrado con **OpenAI**, enfocándose en cinco pilares fundamentales:

1. **Prompting directo** con modelos de lenguaje.  
2. **ChatMessages (system, user, assistant)** y cómo estructuran conversaciones.  
3. **PromptTemplates** para separar lógica, estructura y contenido.  
4. **Chains**: flujos encadenados de entrada → modelo → salida.  
5. **Mini-RAG (Retrieval-Augmented Generation)** con documentos externos.

La práctica sigue la misma línea pedagógica de las anteriores:  
**explicación conceptual + código + interpretación del resultado real**, usando el archivo `14_langchain_openai_intro.py` como base.

---

## 🎯 Objetivos

- Comprender el flujo básico de interacción con OpenAI vía LangChain.  
- Crear y usar **prompts estructurados** y **mensajes en formato chat**.  
- Construir **PromptTemplates** reutilizables.  
- Ejecutar **LLMChains** para automatizar interacción.  
- Generar **structured outputs** en formato JSON.  
- Implementar un **Mini-RAG**: carga de textos, embeddings y retrieval.  
- Interpretar resultados y entender su impacto en aplicaciones reales.

---

# ⚙️ Paso 1 — Configuración Inicial

El archivo carga:

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import HumanMessage, SystemMessage
```

Se establece la **clave de API** y se inicializa un modelo:

```python
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
```

---

## 🔍 Interpretación

- `temperature=0.3` genera respuestas **consistentes, poco aleatorias**, ideal para tareas controladas.  
- Usamos `ChatOpenAI` porque permite trabajar con **mensajes** y no solo cadenas de texto.  

---

# 🧩 Paso 2 — Primer Prompt Simple

Código en  archivo:

```python
respuesta = llm.invoke("Definí 'Transformer' en una sola oración..")
print(respuesta.content)
```

### 📄 Resultado esperado:

```
Un Transformer es una arquitectura de redes neuronales que utiliza mecanismos de atención, especialmente la atención sobre toda la secuencia (self-attention), para procesar entradas en paralelo y capturar dependencias a largo plazo sin recurrencia, lo que la hace especialmente eficaz para el procesamiento de lenguaje natural y otras tareas.

```

---

# 🧩 Paso 3 — Mensajes Tipo Chat

Código del archivo:

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "Sos un asistente conciso, exacto y profesional."),
    ("human", "Explicá {tema} en <= 3 oraciones, con un ejemplo real."),
    ("human", "Por ejemplo, cuando hablamos de 'atención multi-cabeza', se refiere a un mecanismo donde múltiples 'cabezas' o capas de atención permiten que el modelo enfoque diferentes partes de la entrada simultáneamente, mejorando su capacidad de procesamiento de información.")
])

chain = prompt | llm

print(chain.invoke({"tema": "atención multi-cabeza"}).content)
```

### Resultado :

```
La atención multi-cabeza realiza varias atenciones paralelas (cabezas) sobre la misma entrada y luego concatena sus salidas para obtener una representación más rica. Cada cabeza puede enfocarse en un aspecto distinto, como relaciones gramaticales, dependencias a larga distancia o contexto local. Ejemplo real: en el Transformer original (Vaswani et al., 2017), múltiples cabezas de atención permiten al modelo alinear diferentes palabras durante la traducción (una cabeza vincula sujeto con verbo, otra verbo con objeto), mejorando la coherencia frente a una única atención.

```

---

# 🧍 Interpretación

- El **SystemMessage** define el rol del modelo → *experto*.  
- El **HumanMessage** es la consulta.  
- Óptimo para agentes, chatbots y flujos conversacionales.

---

# 🧩 Paso 4 — PromptTemplate

```python
template = PromptTemplate(
    input_variables=["tema"],
    template="Explica {tema} como si fueras profesor de secundaria."
)
chain = LLMChain(llm=llm, prompt=template)
respuesta = chain.invoke({"tema": "los algoritmos"})
```

### Resultado esperado:

```
Un algoritmo es un conjunto de pasos ordenados que permiten resolver un problema...
```

---

# 🧩 Paso 5 — Structured Output (JSON)

```python
json_prompt = "Devuelve la siguiente información en formato JSON:\n- resumen\n- dificultad (1 a 5)\n- tema_principal"
mensaje = HumanMessage(json_prompt)
respuesta = llm.invoke([mensaje])
```

### Ejemplo esperado:

```json
{
  "resumen": "El texto explica qué es un algoritmo...",
  "dificultad": 2,
  "tema_principal": "conceptos básicos de computación"
}
```

---

## 🧍 Interpretación

- **Structured Output** es esencial para pipelines, APIs y extracción.  
- Permite validar formato y disminuir errores.

---

# 🧩 Paso 6 — Chains con contexto

```python
plantilla = PromptTemplate(
    input_variables=["pregunta", "contexto"],
    template="Contexto: {contexto}\nPregunta: {pregunta}\nRespuesta:"
)
respuesta = chain.invoke({
    "pregunta": "¿Qué es Python?",
    "contexto": "Python es un lenguaje de programación popular."
})
```

### Resultado esperado:

```
Python es un lenguaje de programación interpretado y versátil...
```

---

# 🧍 Interpretación

Esto introduce el concepto base del **RAG**:  
**Contexto + Pregunta → Respuesta fundamentada**.

---

# 📚 Paso 7 — Mini-RAG

Código del archivo:

```python
loader = TextLoader("documento.txt")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_documents(docs)

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(chunks, embeddings)

retriever = db.as_retriever()
contexto = retriever.get_relevant_documents("¿Cuál es la idea principal del texto?")
```

### Resultado esperado:

```
El documento trata sobre la importancia del pensamiento algorítmico...
```

---

# 🔍 Interpretación

El Mini-RAG permite:

- usar tus propios documentos  
- actualizar conocimiento sin reentrenar  
- entregar respuestas verificables

---

# 📈 Paso 8 — Análisis General

### 🔹 Prompting  
Respuesta inmediata, pero poco estructurada.

### 🔹 Templates  
Separan lógica y contenido → profesional y escalable.

### 🔹 Structured Output  
Ideal para extracción y análisis de datos.

### 🔹 Chains  
Automatizan flujos completos.

### 🔹 RAG  
Convierte LLMs en sistemas **contextuales** basados en documentos reales.

---

# 🧠 Conclusiones

1. LangChain facilita el uso profesional de LLMs mediante plantillas, cadenas y mensajes.  
2. Structured Output mejora la robustez y la integración del sistema.  
3. El Mini-RAG demostró cómo extender la capacidad del modelo con información externa.  
4. La práctica completa sienta bases para agentes, memorias y pipelines avanzados.

---

# 🤔 Preguntas de Reflexión

- ¿Por qué es importante estructurar la conversación con mensajes System/User?  
- ¿Qué ventajas tiene RAG sobre prompting simple?  
- ¿En qué escenarios es obligatorio usar JSON estructurado?  
- ¿Qué mejoras agregarías a tu propio sistema RAG?

---

# 📁 Evidencias


- 📓 Notebook ejecutado: [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1neuic9mh5V_S7mYNaR3bUfqxLZtvBDVR?usp=sharing)


- prompting_basico.png  
- chat_messages.png  
- prompt_template.png  
- structured_output.png  
- chain_contexto.png  
- mini_rag_diagrama.png  

---

# 📥 Fin del documento
