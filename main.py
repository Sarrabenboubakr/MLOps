"""Point d'entrée : exécute le pipeline ML via des arguments CLI."""

import argparse
from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)


def main():
    parser = argparse.ArgumentParser(description="Pipeline ML Churn")
    parser.add_argument(
        "--data", default="Churn_Modelling.csv", help="Chemin du fichier CSV"
    )
    parser.add_argument(
        "--model", choices=["logistic", "rf"], default="logistic", help="Type de modèle"
    )
    parser.add_argument(
        "--mode", choices=["train", "load"], default="train", help="Mode d'exécution"
    )
    args = parser.parse_args()

    if args.mode == "train":
        print(f"--- 1. Préparation des données depuis {args.data} ---")
        X_train, X_test, y_train, y_test, scaler = prepare_data(args.data)

        print(f"--- 2. Entraînement du modèle ({args.model}) ---")
        model = train_model(X_train, y_train, model_type=args.model)

        print("--- 3. Évaluation du modèle ---")
        evaluate_model(model, X_test, y_test)

        print("--- 4. Sauvegarde du modèle ---")
        save_model(model, scaler)

    else:
        print("--- Mode : Chargement du modèle existant ---")
        try:
            model, scaler = load_model()
            X_train, X_test, y_train, y_test, _ = prepare_data(args.data)
            evaluate_model(model, X_test, y_test)
        except FileNotFoundError:
            print(
                "Erreur : Aucun modèle trouvé dans le dossier 'models/'. Lancez d'abord le mode 'train'."
            )


if __name__ == "__main__":
    main()
