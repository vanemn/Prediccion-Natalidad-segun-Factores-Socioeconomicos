# 🧠 Predicción de Natalidad según Factores Socioeconómicos

Este proyecto fue desarrollado como parte de una evaluación final en el contexto de investigación demográfica. El objetivo principal fue construir un modelo predictivo basado en redes neuronales para estimar la tasa de natalidad en distintos países, utilizando variables socioeconómicas como el PIB per cápita, acceso a salud, educación, empleo femenino, urbanización y edad promedio de maternidad.

---

## 🎯 Objetivos del Proyecto

- Diseñar y entrenar una red neuronal para resolver un problema de regresión.
- Aplicar funciones de activación, optimizadores y estrategias de regularización para evitar sobreajuste.
- Evaluar múltiples configuraciones del modelo y seleccionar la más precisa.
- Analizar la influencia de cada variable en la predicción y extraer conclusiones sobre patrones demográficos globales.

---

## 🛠️ Tecnologías y Herramientas

- Python 3.10  
- Pandas, NumPy, Matplotlib, Seaborn  
- Scikit-learn, TensorFlow (Keras)  
- Técnicas aplicadas:  
  - Regularización (Dropout, L2)  
  - Permutation Importance  
  - Regresión lineal interpretativa  
  - Visualización de efectos marginales  

---

## 📁 Estructura del Proyecto
--- ´´´
├── consolidado.py                  # Script principal con carga, modelado y evaluación
├── dataset_natalidad.csv          # Dataset con variables socioeconómicas por país
├── resultados/
│   ├── 2_resultados_configs.csv   # Métricas por configuración
│   ├── 3_metricas_finales.json    # Métricas del mejor modelo
│   ├── 3_predicciones_test.csv    # Predicciones vs valores reales
│   ├── 4_importancia_permutacion.csv  # Importancia de variables (permutación)
│   ├── 4_coeficientes_lineal.csv      # Coeficientes del modelo lineal
│   └── 5_reflexion_operativa.json     # Reflexión final y mejoras propuestas
├── figuras/
│   ├── 1_correlacion.png
│   ├── 1_distribuciones.png
│   ├── 2_curvas_entrenamiento.png
│   ├── 3_real_vs_predicho.png
│   ├── 3_residuales.png
│   ├── 4_importancia_permutacion.png
│   ├── 4_coefs_lineal.png
│   └── 4_parcial_<variable>.png

--- ´´´
## 🔍 Metodología

### 1. Carga y exploración de datos
- Análisis de correlaciones entre variables.
- Visualización de distribuciones numéricas.
- Preparación de datos para modelado (escalado, separación de conjuntos).

### 2. Diseño y entrenamiento del modelo
- Red neuronal con:
  - Capa de entrada según número de variables predictoras.
  - Mínimo 2 capas ocultas con activaciones `relu`, `tanh` y `selu`.
  - Regularización con `Dropout` y `L2`.
  - Optimización con `Adam` y `RMSprop`.
- Comparación de 5 configuraciones distintas.

### 3. Evaluación y análisis
- Métricas: MAE, RMSE, R².
- Importancia de variables mediante permutación.
- Interpretabilidad con regresión lineal.
- Visualización de efectos marginales para las variables más influyentes.

### 4. Reflexión final
- Identificación de variables clave: educación, urbanización, edad de maternidad.
- Relación con tendencias demográficas globales.
- Propuestas de mejora: modelos alternativos, explicabilidad avanzada (SHAP), series temporales.

---

## 📊 Resultados Destacados


| Métrica        | Valor     | Interpretación                                                                 |
|----------------|-----------|--------------------------------------------------------------------------------|
| MAE (Error Absoluto Medio) | 4.84      | En promedio, el modelo se equivoca en ~4.84 unidades de natalidad.         |
| RMSE (Raíz del Error Cuadrático Medio) | 5.72      | Penaliza más los errores grandes. Buen indicador de precisión general.     |
| R² (Coeficiente de Determinación)      | 0.545     | El modelo explica el 54.5% de la variabilidad en la tasa de natalidad.     |
| Mejor Configuración                    | relu + adam + lr=0.001 + dropout=0.3 | Combinación óptima para este dataset. |
| Variables más influyentes              | PIB_per_capita, Urbanización, Empleo Femenino | Factores clave en la predicción. |
> Las variables más influyentes fueron aquellas relacionadas con educación, urbanización y edad promedio de maternidad, alineadas con patrones globales de natalidad.

---

## 🚀 Propuestas de Mejora

- Incorporar datos longitudinales y políticas públicas.
- Validación cruzada y búsqueda bayesiana de hiperparámetros.
- Comparación con modelos de árbol (XGBoost, RandomForest).
- Aplicar técnicas de explicabilidad como SHAP y estimación de incertidumbre.

---

## 👩‍💻 Autora

**Vanessa Morales Norambuena **  
Creadora y estratega digital con formación en ciencia de datos, machine learning aplicado y gestión de proyectos.  
Especializada gestión de proyectos.

---

## 📌 Artefactos Generados

- Visualizaciones: correlaciones, distribuciones, curvas de entrenamiento, residuales, efectos marginales.
- Métricas exportables en `.csv` y `.json`.
- Modelos entrenados y guardados en formato `.keras`.
- Reflexión operativa documentada para transferencia grupal.

