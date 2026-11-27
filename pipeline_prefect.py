"""Pipeline ML Churn orchestré avec Prefect."""

import argparse
import subprocess
from pathlib import Path

from prefect import task, flow

import model_pipeline as core

PROJECT_DIR = Path(__file__).parent
PY_FILES = ["model_pipeline.py", "main.py", "pipeline_prefect.py"]
DATA_PATH = "Churn_Modelling.csv"


@task
def install_dependencies():
    """Installe les dépendances depuis requirements.txt."""
    subprocess.run(["pip", "install", "-r", "requirements.txt"], check=True)


@task
def prepare_data():
    """Charge et prépare les données."""
    return core.prepare_data(DATA_PATH)


@task
def train_model(data):
    """Entraîne un modèle sur les données préparées."""
    x_train, _x_test, y_train, _y_test, _scaler = data
    return core.train_model(x_train, y_train, model_type="logistic")


@task
def save_model(model_and_scaler):
    """Sauvegarde le modèle et le scaler sur disque."""
    model, scaler = model_and_scaler
    core.save_model(model, scaler)


@task
def load_model():
    """Charge le modèle et le scaler depuis le disque."""
    return core.load_model()


@task
def evaluate_model(model_and_data):
    """Évalue le modèle sur les données de test."""
    model, data = model_and_data
    _x_train, x_test, _y_train, y_test, _scaler = data
    return core.evaluate_model(model, x_test, y_test)


@task
def format_code():
    """Formate le code avec black."""
    subprocess.run(["black", *PY_FILES], cwd=PROJECT_DIR, check=True)


@task
def check_quality():
    """Analyse la qualité avec pylint (seuil 5)."""
    subprocess.run(
        ["pylint", "--fail-under=5", *PY_FILES],
        cwd=PROJECT_DIR,
        check=True,
    )


@task
def check_security():
    """Analyse la sécurité avec bandit."""
    subprocess.run(["bandit", "-r", *PY_FILES], cwd=PROJECT_DIR, check=False)


@task
def run_tests():
    """Exécute les tests unitaires."""
    subprocess.run(["pytest", "tests/"], cwd=PROJECT_DIR, check=True)


@flow(name="code")
def code_flow():
    """Flow de qualité du code."""
    format_code()
    check_quality()
    check_security()
    run_tests()


@flow(name="train")
def train_flow():
    """Flow d'entraînement."""
    data = prepare_data()
    model = train_model(data)
    _x_train, _x_test, _y_train, _y_test, scaler = data
    save_model((model, scaler))


@flow(name="evaluate")
def evaluate_flow():
    """Flow d'évaluation."""
    loaded = load_model()
    model = loaded[0] if isinstance(loaded, tuple) else loaded
    data = prepare_data()
    evaluate_model((model, data))


@flow(name="all")
def all_flow():
    """Flow complet."""
    install_dependencies()
    code_flow()
    data = prepare_data()
    model = train_model(data)
    _x_train, _x_test, _y_train, _y_test, scaler = data
    save_model((model, scaler))
    evaluate_model((model, data))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--flow",
        required=True,
        choices=["all", "code", "train", "evaluate"],
    )
    args = parser.parse_args()

    {
        "all": all_flow,
        "code": code_flow,
        "train": train_flow,
        "evaluate": evaluate_flow,
    }[args.flow]()
