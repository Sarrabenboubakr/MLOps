"""Tests unitaires minimaux pour le pipeline ML."""

from pathlib import Path
from model_pipeline import prepare_data


DATA_PATH = Path(__file__).parent.parent / "Churn_Modelling.csv"


def test_prepare_data():
    """Vérifie que prepare_data renvoie bien 5 objets."""
    data = prepare_data(str(DATA_PATH))
    assert data is not None
    assert len(data) == 5
