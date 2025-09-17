# -*- coding: utf-8 -*-
# consolidado.py
# EVALUACIÓN FINAL: PREDICCIÓN DE NATALIDAD SEGÚN FACTORES SOCIOECONÓMICOS

# Cada requerimiento está señalado con comentarios [Punto X] y bullets del enunciado.

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# =============================================================================
# [Punto 1 - Carga y exploración de datos (1 punto)]
# • Cargar dataset con variables indicadas.
# • Analizar correlaciones y visualizar distribuciones.
# =============================================================================

DATA_PATH = "dataset_natalidad.csv"  # Cambia a la ruta real de tu dataset

print("Cargando datos desde:", DATA_PATH)
data = pd.read_csv(DATA_PATH)
print("Datos cargados. Shape:", data.shape)
print("\nColumnas disponibles:", list(data.columns))
print("\nPrimeras 5 filas:\n", data.head())
print("\nEstadísticas descriptivas:\n", data.describe())

os.makedirs("figuras", exist_ok=True)
num_cols = data.select_dtypes(include=[np.number]).columns

# Matriz de correlación
plt.figure(figsize=(10, 8))
sns.heatmap(data[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matriz de correlación")
plt.tight_layout()
plt.savefig("figuras/1_correlacion.png", dpi=150)
plt.close()

# Distribuciones de variables numéricas
data[num_cols].hist(figsize=(12, 8), bins=12)
plt.suptitle("Distribuciones de variables numéricas", y=1.02)
plt.tight_layout()
plt.savefig("figuras/1_distribuciones.png", dpi=150)
plt.close()

# =============================================================================
# Preparación de datos para modelado
# =============================================================================
TARGET = "Tasa_Natalidad"
ID_COLS = [c for c in ["País", "Pais", "Country"] if c in data.columns]

X_df = data.drop(columns=[TARGET] + ID_COLS)
y = data[TARGET].values

feature_names = list(X_df.columns)
X = X_df.values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# =============================================================================
# [Punto 2 - Diseño y entrenamiento del modelo (5 puntos)]
# • Estructura NN:
#   o Capa de entrada con tantas neuronas como variables predictoras.
#   o Mínimo de 2 capas ocultas con activaciones adecuadas.
#   o Capa de salida con 1 neurona para predecir la tasa de natalidad.
# • Probar diferentes funciones de activación y evaluar su impacto.
# • Usar optimizadores y explorar distintos learning rates.
# • Regularización (Dropout o L2) para evitar sobreajuste.
# • Entrenar con función de pérdida para regresión (MSE/MAE).
# =============================================================================

def build_model(input_dim,
                activation="relu",            # [Punto 2] • Función de activación
                optimizer_name="adam",        # [Punto 2] • Optimizer
                lr=1e-3,                      # [Punto 2] • Learning rate
                dropout_rate=0.3,             # [Punto 2] • Regularización Dropout
                use_l2=False,                 # [Punto 2] • Regularización L2
                l2_lambda=1e-3,
                width=64):
    reg = l2(l2_lambda) if use_l2 else None

    model = Sequential()
    # [Punto 2] • Capa de entrada explícita
    model.add(Input(shape=(input_dim,)))
    # [Punto 2] • Capa de entrada: tantas neuronas como variables predictoras (input_dim)
    # Primera capa oculta (actúa como capa de entrada al definir input_dim)
    model.add(Dense(width, input_dim=input_dim, activation=activation, kernel_regularizer=reg))

    # [Punto 2] • Regularización con Dropout
    model.add(Dropout(dropout_rate))

    # [Punto 2] • Segunda capa oculta (mínimo 2 capas ocultas)
    # [Punto 2] • Primera capa oculta
    model.add(Dense(width, activation=activation, kernel_regularizer=reg))
    model.add(Dropout(dropout_rate))
    # [Punto 2] • Segunda capa oculta
    model.add(Dense(width // 2, activation=activation, kernel_regularizer=reg))
    model.add(Dropout(dropout_rate / 2))
    # [Punto 2] • Capa de salida con una neurona (regresión)
    model.add(Dense(1, activation=None))

    # [Punto 2] • Optimizer + learning rate
    if optimizer_name.lower() == "adam":
        opt = Adam(learning_rate=lr)
    elif optimizer_name.lower() == "rmsprop":
        opt = RMSprop(learning_rate=lr)
    else:
        opt = Adam(learning_rate=lr)

    # [Punto 2] • Función de pérdida para regresión y métrica
    model.compile(optimizer=opt, loss="mse", metrics=["mae"])
    return model

# Callbacks para prevenir sobreajuste y estabilizar el entrenamiento
# [Punto 2] • Estrategias anti-overfitting: EarlyStopping y ReduceLROnPlateau
callbacks = [
    EarlyStopping(monitor="val_mae", patience=20, restore_best_weights=True, verbose=0),
    ReduceLROnPlateau(monitor="val_mae", factor=0.5, patience=10, min_lr=1e-6, verbose=0),
]

# Configuraciones a comparar
# [Punto 2] • Variar activaciones, optimizadores, learning rate y regularización
configs = [
    {"activation": "relu", "opt": "adam",    "lr": 1e-3, "dropout": 0.3, "use_l2": False},
    {"activation": "tanh", "opt": "adam",    "lr": 1e-3, "dropout": 0.3, "use_l2": False},
    {"activation": "relu", "opt": "adam",    "lr": 5e-4, "dropout": 0.4, "use_l2": True},
    {"activation": "tanh", "opt": "rmsprop", "lr": 1e-3, "dropout": 0.2, "use_l2": True},
    {"activation": "selu", "opt": "adam",    "lr": 1e-3, "dropout": 0.2, "use_l2": False},
]

os.makedirs("modelos", exist_ok=True)
os.makedirs("resultados", exist_ok=True)

results = []
history_dict = {}

for i, cfg in enumerate(configs, 1):
    print(f"\nEntrenando configuración {i}/{len(configs)}:", cfg)

    model = build_model(
        input_dim=X_train_s.shape[1],
        activation=cfg["activation"],
        optimizer_name=cfg["opt"],
        lr=cfg["lr"],
        dropout_rate=cfg["dropout"],
        use_l2=cfg["use_l2"],
        l2_lambda=1e-3,
        width=64
    )

    # [Punto 2] • Entrenamiento con validación y callbacks anti-overfitting
    ckpt_path = f"modelos/mejor_modelo_cfg{i}.keras"
    ckpt = ModelCheckpoint(ckpt_path, monitor="val_mae", save_best_only=True, verbose=0)

    hist = model.fit(
        X_train_s, y_train,
        validation_split=0.2,
        epochs=400,
        batch_size=16,
        callbacks=callbacks + [ckpt],
        verbose=0
    )
    history_dict[i] = hist.history

    # =============================================================================
    # [Punto 3 - Evaluación y optimización del modelo (3 puntos)]
    # • Evaluar con datos de prueba (test).
    # • Ajustar hiperparámetros (se hace al comparar múltiples configuraciones).
    # =============================================================================

    # Cargar mejores pesos y evaluar en test
    model.load_weights(ckpt_path)
    y_pred = model.predict(X_test_s, verbose=0).ravel()

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # 
    r2 = r2_score(y_test, y_pred)

    results.append({
        "config_id": i,
        "activation": cfg["activation"],
        "optimizer": cfg["opt"],
        "lr": cfg["lr"],
        "dropout": cfg["dropout"],
        "use_l2": cfg["use_l2"],
        "test_mae": mae,
        "test_rmse": rmse,
        "test_r2": r2,
        "ckpt": ckpt_path
    })
    print(f"Config {i} -> MAE: {mae:.3f} | RMSE: {rmse:.3f} | R2: {r2:.3f}")

results_df = pd.DataFrame(results).sort_values(by="test_mae")
results_df.to_csv("resultados/2_resultados_configs.csv", index=False)
print("\nResultados ordenados por MAE:\n", results_df)

# Selección del mejor modelo
best = results_df.iloc[0].to_dict()
print("\nMejor configuración:", best)

best_model = build_model(
    input_dim=X_train_s.shape[1],
    activation=best["activation"],
    optimizer_name=best["optimizer"],
    lr=best["lr"],
    dropout_rate=best["dropout"],
    use_l2=best["use_l2"],
    l2_lambda=1e-3,
    width=64
)
best_model.load_weights(best["ckpt"])
y_pred_test = best_model.predict(X_test_s, verbose=0).ravel()

# Curvas de entrenamiento para la mejor config
hist_best = history_dict[int(best["config_id"])]
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(hist_best["loss"], label="train_mse")
plt.plot(hist_best["val_loss"], label="val_mse")
plt.title("Pérdida (MSE)")
plt.legend()

plt.subplot(1,2,2)
plt.plot(hist_best["mae"], label="train_mae")
plt.plot(hist_best["val_mae"], label="val_mae")
plt.title("MAE")
plt.legend()
plt.tight_layout()
plt.savefig("figuras/2_curvas_entrenamiento.png", dpi=150)
plt.close()

# Real vs Predicho (test)
plt.figure(figsize=(5,5))
plt.scatter(y_test, y_pred_test, alpha=0.85)
minv, maxv = y_test.min(), y_test.max()
plt.plot([minv, maxv], [minv, maxv], "r--")
plt.xlabel("Real")
plt.ylabel("Predicho")
plt.title("Real vs Predicho (Test)")
plt.tight_layout()
plt.savefig("figuras/3_real_vs_predicho.png", dpi=150)
plt.close()

# Residuales
residuals = y_test - y_pred_test
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
sns.histplot(residuals, kde=True)
plt.title("Distribución de residuales")
plt.subplot(1,2,2)
plt.scatter(y_pred_test, residuals, alpha=0.85)
plt.axhline(0, color="r", linestyle="--")
plt.xlabel("Predicho")
plt.ylabel("Residual")
plt.title("Residuales vs Predicho")
plt.tight_layout()
plt.savefig("figuras/3_residuales.png", dpi=150)
plt.close()

# Guardar métricas finales de la mejor configuración
final_metrics = {
    "best_config": {
        "activation": best["activation"],
        "optimizer": best["optimizer"],
        "lr": float(best["lr"]),
        "dropout": float(best["dropout"]),
        "use_l2": bool(best["use_l2"])
    },
    "test_mae": float(results_df.iloc[0]["test_mae"]),
    "test_rmse": float(results_df.iloc[0]["test_rmse"]),
    "test_r2": float(results_df.iloc[0]["test_r2"])
}
with open("resultados/3_metricas_finales.json", "w", encoding="utf-8") as f:
    json.dump(final_metrics, f, ensure_ascii=False, indent=2)

# =============================================================================
# [Punto 3 - Análisis del impacto de cada variable]
# • Analizar la influencia de cada variable en la predicción.
#   Implementado con Importancia por Permutación + baseline lineal.
# =============================================================================

# Importancia por permutación (delta MAE al permutar la variable)
rng = np.random.default_rng(42)
base_mae = mean_absolute_error(y_test, y_pred_test)
perm_importance = []
X_test_s_base = X_test_s.copy()

for j, name in enumerate(feature_names):
    X_perm = X_test_s_base.copy()
    X_perm[:, j] = rng.permutation(X_perm[:, j])
    y_perm_pred = best_model.predict(X_perm, verbose=0).ravel()
    mae_perm = mean_absolute_error(y_test, y_perm_pred)
    delta = mae_perm - base_mae
    perm_importance.append((name, delta))

perm_df = pd.DataFrame(perm_importance, columns=["variable", "delta_mae"]).sort_values("delta_mae", ascending=False)
perm_df.to_csv("resultados/4_importancia_permutacion.csv", index=False)
print("\nImportancia por permutación (delta MAE):\n", perm_df)

plt.figure(figsize=(8,5))
sns.barplot(data=perm_df, x="delta_mae", y="variable", palette="viridis")
plt.xlabel("Aumento de MAE al permutar (más alto = más importante)")
plt.ylabel("Variable")
plt.title("Importancia de variables por permutación (Test)")
plt.tight_layout()
plt.savefig("figuras/4_importancia_permutacion.png", dpi=150)
plt.close()

# Baseline interpretable: Regresión Lineal sobre datos escalados (signo y magnitud)
linreg = LinearRegression().fit(X_train_s, y_train)
coefs = pd.Series(linreg.coef_, index=feature_names).sort_values()
coefs.to_csv("resultados/4_coeficientes_lineal.csv")
print("\nCoeficientes regresión lineal:\n", coefs)

plt.figure(figsize=(8,5))
colors = ["#4C72B0" if v > 0 else "#DD8452" for v in coefs.values]
coefs.plot(kind="barh", color=colors)
plt.title("Coeficientes modelo lineal (X escalado)")
plt.xlabel("Coeficiente")
plt.tight_layout()
plt.savefig("figuras/4_coefs_lineal.png", dpi=150)
plt.close()

# Curvas marginales simples para top-3 variables por permutación
top_vars = perm_df["variable"].head(3).tolist()

def partial_plot(var, n_points=25):
    xi = feature_names.index(var)
    grid = np.linspace(X_train_s[:, xi].min(), X_train_s[:, xi].max(), n_points)
    X_ref = np.median(X_test_s, axis=0).reshape(1, -1)
    preds = []
    for g in grid:
        X_tmp = X_ref.copy()
        X_tmp[0, xi] = g
        preds.append(best_model.predict(X_tmp, verbose=0).ravel()[0])

    mean_ = scaler.mean_[xi]
    std_ = np.sqrt(scaler.var_[xi])
    grid_orig = grid * std_ + mean_

    plt.figure(figsize=(5,4))
    plt.plot(grid_orig, preds, "-o")
    plt.title(f"Efecto parcial aproximado: {var}")
    plt.xlabel(var)
    plt.ylabel("Tasa de natalidad predicha")
    plt.tight_layout()
    plt.savefig(f"figuras/4_parcial_{var}.png", dpi=150)
    plt.close()

for v in top_vars:
    partial_plot(v)

# =============================================================================
# [Punto 3 - Realizar predicciones y comparar con datos reales]
# =============================================================================
pred_df = pd.DataFrame({"y_real": y_test, "y_pred": y_pred_test})
pred_df.to_csv("resultados/3_predicciones_test.csv", index=False)

# =============================================================================
# [Punto 4 - Análisis de resultados y reflexión final (1 punto)]
# • Explicar variables más influyentes (usa perm_df y coefs).
# • Relacionar con tendencias demográficas globales.
# • Proponer mejoras para futuras versiones.
# Nota: aquí dejamos un resumen operativo impreso y guardado.
# =============================================================================

reflexion = {
    "variables_mas_influyentes_top3": perm_df.iloc[:3].to_dict(orient="records"),
    "lectura_signo_baseline_lineal": "Coeficientes positivos aumentan la predicción; negativos la reducen (ver resultados/4_coeficientes_lineal.csv).",
    "tendencias_globales": "Mayor educación, urbanización, edad de maternidad y acceso a salud suelen asociarse a menor natalidad; contrastar con tu ranking empírico.",
    "mejoras": [
        "Añadir datos longitudinales (series temporales) y variables de políticas públicas.",
        "Validación cruzada y búsqueda bayesiana de hiperparámetros.",
        "Comparar con modelos de árbol (XGBoost, RandomForest) y ensamblados.",
        "Explicabilidad avanzada (SHAP) y estimación de incertidumbre (intervalos)."
    ]
}
with open("resultados/5_reflexion_operativa.json", "w", encoding="utf-8") as f:
    json.dump(reflexion, f, ensure_ascii=False, indent=2)

print("\n--- RESUMEN FINAL ---")
print("Mejor config:", final_metrics["best_config"])
print("MAE Test:", final_metrics["test_mae"])
print("RMSE Test:", final_metrics["test_rmse"])
print("R2 Test:", final_metrics["test_r2"])
print("\nTop-3 variables por importancia (permutación):")
print(perm_df.head(3))
print("\nArtefactos guardados en: figuras/, resultados/, modelos/")
