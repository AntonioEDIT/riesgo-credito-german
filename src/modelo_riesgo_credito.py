"""
Modelo de riesgo de crédito (probabilidad de impago) con Python.

Pasos:
  1. Carga de datos (German Credit, 1,000 solicitudes de crédito)
  2. Análisis exploratorio (EDA) básico
  3. Preprocesamiento (one-hot encoding y escalado)
  4. Entrenamiento de 3 modelos: Regresión Logística, Árbol de Decisión y Random Forest
  5. Evaluación con métricas de clasificación y de riesgo de crédito (AUC, KS, Gini)
  6. Gráficas y archivos de resultados

Ejecutar desde la carpeta raíz del proyecto:
    python src/modelo_riesgo_credito.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # guarda las gráficas sin abrir ventanas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ks_2samp
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parent.parent
RUTA_DATOS = RAIZ / "data" / "german_credit.csv"
URL_DATOS = "https://raw.githubusercontent.com/selva86/datasets/master/GermanCredit.csv"
CARPETA_IMG = RAIZ / "images"
CARPETA_OUT = RAIZ / "outputs"
SEMILLA = 42

CARPETA_IMG.mkdir(exist_ok=True)
CARPETA_OUT.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid")


# ---------------------------------------------------------------------------
# 1. Carga de datos
# ---------------------------------------------------------------------------
def cargar_datos() -> pd.DataFrame:
    """Lee el CSV local; si no existe, lo descarga y lo guarda en data/."""
    if RUTA_DATOS.exists():
        df = pd.read_csv(RUTA_DATOS)
    else:
        print("Descargando datos...")
        df = pd.read_csv(URL_DATOS)
        RUTA_DATOS.parent.mkdir(exist_ok=True)
        df.to_csv(RUTA_DATOS, index=False)

    # En el archivo original: credit_risk = 1 (buen pagador), 0 (mal pagador).
    # Creamos la variable objetivo "impago": 1 = mal pagador (el evento que queremos predecir).
    df["impago"] = (df["credit_risk"] == 0).astype(int)
    return df.drop(columns="credit_risk")


# ---------------------------------------------------------------------------
# 2. Análisis exploratorio (EDA)
# ---------------------------------------------------------------------------
def eda(df: pd.DataFrame) -> None:
    print("\n=== 2. ANÁLISIS EXPLORATORIO ===")
    print(f"Filas: {df.shape[0]} | Columnas: {df.shape[1]}")
    print(f"Valores nulos: {df.isna().sum().sum()}")
    print(f"Duplicados: {df.duplicated().sum()}")
    print(f"Tasa de impago: {df['impago'].mean():.1%}")
    print("\nEstadísticos de variables numéricas clave:")
    print(df[["duration", "amount", "age"]].describe().round(1))

    # Gráfica 1: tasa de impago por historial crediticio
    tasa = (
        df.groupby("credit_history")["impago"].mean().sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.barplot(x=tasa.values * 100, y=tasa.index, ax=ax, color="#3b6ea5")
    ax.set_xlabel("Tasa de impago (%)")
    ax.set_ylabel("")
    ax.set_title("Tasa de impago por historial crediticio")
    fig.tight_layout()
    fig.savefig(CARPETA_IMG / "eda_impago_por_historial.png", dpi=150)
    plt.close(fig)

    # Gráfica 2: monto del crédito según si hubo impago
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.boxplot(data=df, x="impago", y="amount", ax=ax, color="#9cc3e6")
    ax.set_xticks([0, 1], ["Pagó (0)", "Impago (1)"])
    ax.set_xlabel("")
    ax.set_ylabel("Monto del crédito (DM)")
    ax.set_title("Monto del crédito vs. impago")
    fig.tight_layout()
    fig.savefig(CARPETA_IMG / "eda_monto_vs_impago.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Preprocesamiento
# ---------------------------------------------------------------------------
def crear_preprocesador(X: pd.DataFrame) -> ColumnTransformer:
    numericas = X.select_dtypes(include="number").columns.tolist()
    categoricas = X.select_dtypes(exclude="number").columns.tolist()
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numericas),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categoricas),
        ]
    )


# ---------------------------------------------------------------------------
# 5. Métricas
# ---------------------------------------------------------------------------
def estadistico_ks(y_real, prob) -> float:
    """KS: máxima separación entre las distribuciones de score de buenos y malos."""
    prob = np.asarray(prob)
    y_real = np.asarray(y_real)
    return ks_2samp(prob[y_real == 1], prob[y_real == 0]).statistic


def evaluar(nombre, y_real, prob, umbral=0.5) -> dict:
    pred = (prob >= umbral).astype(int)
    auc = roc_auc_score(y_real, prob)
    return {
        "modelo": nombre,
        "accuracy": accuracy_score(y_real, pred),
        "precision": precision_score(y_real, pred, zero_division=0),
        "recall": recall_score(y_real, pred),
        "f1": f1_score(y_real, pred),
        "auc_roc": auc,
        "gini": 2 * auc - 1,
        "ks": estadistico_ks(y_real, prob),
    }


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------
def main() -> None:
    print("=== 1. CARGA DE DATOS ===")
    df = cargar_datos()
    print(f"Datos cargados: {len(df)} registros")

    eda(df)

    X = df.drop(columns="impago")
    y = df["impago"]

    # División 70/30 estratificada (mantiene la misma proporción de impagos)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=SEMILLA
    )

    print("\n=== 4. ENTRENAMIENTO DE MODELOS ===")
    # class_weight="balanced" compensa que hay menos impagos (30%) que buenos pagadores
    modelos = {
        "Regresión Logística": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=SEMILLA
        ),
        "Árbol de Decisión": DecisionTreeClassifier(
            max_depth=4, min_samples_leaf=20, class_weight="balanced",
            random_state=SEMILLA,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight="balanced",
            random_state=SEMILLA,
        ),
    }

    resultados, probabilidades, pipelines = [], {}, {}
    for nombre, modelo in modelos.items():
        pipe = Pipeline([("prep", crear_preprocesador(X)), ("modelo", modelo)])
        pipe.fit(X_train, y_train)
        prob = pipe.predict_proba(X_test)[:, 1]  # probabilidad de impago
        probabilidades[nombre] = prob
        pipelines[nombre] = pipe
        resultados.append(evaluar(nombre, y_test, prob))
        print(f"  ✔ {nombre} entrenado")

    tabla = pd.DataFrame(resultados).set_index("modelo").round(3)
    tabla.to_csv(CARPETA_OUT / "metricas_modelos.csv")
    print("\n=== 5. MÉTRICAS EN DATOS DE PRUEBA (umbral 0.5) ===")
    print(tabla.to_string())

    mejor = tabla["auc_roc"].idxmax()
    print(f"\nMejor modelo por AUC: {mejor}")

    # --- Gráfica: curvas ROC ---
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for nombre, prob in probabilidades.items():
        fpr, tpr, _ = roc_curve(y_test, prob)
        ax.plot(fpr, tpr, label=f"{nombre} (AUC = {tabla.loc[nombre, 'auc_roc']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Azar (AUC = 0.5)")
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos (recall)")
    ax.set_title("Curvas ROC – comparación de modelos")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(CARPETA_IMG / "curvas_roc.png", dpi=150)
    plt.close(fig)

    # --- Gráfica: matriz de confusión del mejor modelo ---
    pred = (probabilidades[mejor] >= 0.5).astype(int)
    cm = confusion_matrix(y_test, pred)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
        xticklabels=["Pagó", "Impago"], yticklabels=["Pagó", "Impago"],
    )
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de confusión – {mejor}")
    fig.tight_layout()
    fig.savefig(CARPETA_IMG / "matriz_confusion.png", dpi=150)
    plt.close(fig)

    # --- Gráfica: variables más importantes (Random Forest) ---
    rf = pipelines["Random Forest"]
    nombres_vars = rf.named_steps["prep"].get_feature_names_out()
    importancias = pd.Series(
        rf.named_steps["modelo"].feature_importances_, index=nombres_vars
    ).sort_values(ascending=False).head(10)
    importancias.index = [n.split("__", 1)[1] for n in importancias.index]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=importancias.values, y=importancias.index, ax=ax, color="#3b6ea5")
    ax.set_xlabel("Importancia")
    ax.set_ylabel("")
    ax.set_title("Top 10 variables – Random Forest")
    fig.tight_layout()
    fig.savefig(CARPETA_IMG / "importancia_variables.png", dpi=150)
    plt.close(fig)

    # --- Score de riesgo por solicitante (para negocio) ---
    scores = X_test.copy()
    scores["impago_real"] = y_test.values
    scores["prob_impago"] = probabilidades[mejor].round(3)
    scores["segmento_riesgo"] = pd.cut(
        scores["prob_impago"], bins=[0, 0.3, 0.6, 1.0],
        labels=["Bajo", "Medio", "Alto"], include_lowest=True,
    )
    scores.to_csv(CARPETA_OUT / "scores_prueba.csv", index=False)

    resumen = scores.groupby("segmento_riesgo", observed=True).agg(
        solicitantes=("impago_real", "size"),
        tasa_impago_real=("impago_real", "mean"),
    )
    resumen["tasa_impago_real"] = (resumen["tasa_impago_real"] * 100).round(1)
    print("\n=== 6. SEGMENTOS DE RIESGO (datos de prueba) ===")
    print(resumen.to_string())

    print("\nListo. Gráficas en images/ y resultados en outputs/.")


if __name__ == "__main__":
    main()
