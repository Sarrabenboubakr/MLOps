"""Pipeline ML modulaire pour la prédiction de churn (Churn_Modelling.csv)."""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score


def prepare_data(path: str, test_size: float = 0.2, random_state: int = 42):
    """Charge, nettoie et prépare les données.

    Returns:
        X_train, X_test, y_train, y_test, scaler
    """
    df = pd.read_csv(path, encoding="utf-8")

    # --- Nettoyage des artefacts ---
    df = df.dropna(subset=["Exited"])  # supprime la ligne 886 tronquée
    df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])

    # --- Encodage des variables catégorielles ---
    df = pd.get_dummies(df, columns=["Geography", "Gender"], drop_first=True)

    X = df.drop(columns=["Exited"])
    y = df["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler


def train_model(X_train, y_train, model_type: str = "logistic", **kwargs):
    """Entraîne un modèle et le retourne."""
    if model_type == "logistic":
        model = LogisticRegression(max_iter=1000, **kwargs)
    elif model_type == "rf":
        model = RandomForestClassifier(random_state=42, **kwargs)
    else:
        raise ValueError(f"model_type inconnu : {model_type}")

    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Évalue le modèle et retourne un dict de métriques."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_proba),
        "report": classification_report(y_test, y_pred),
    }
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"AUC     : {metrics['auc']:.4f}")
    print(metrics["report"])
    return metrics


def save_model(model, scaler, out_dir: str = "models"):
    """Sauvegarde le modèle et le scaler avec joblib."""
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(model, os.path.join(out_dir, "model.joblib"))
    joblib.dump(scaler, os.path.join(out_dir, "scaler.joblib"))
    print(f"Modèle et scaler sauvegardés dans {out_dir}/")


def load_model(in_dir: str = "models"):
    """Charge le modèle et le scaler sauvegardés."""
    model = joblib.load(os.path.join(in_dir, "model.joblib"))
    scaler = joblib.load(os.path.join(in_dir, "scaler.joblib"))
    return model, scaler
