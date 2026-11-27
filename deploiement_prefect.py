"""Enregistre les flows comme déploiements Prefect planifiés."""

from prefect.schedules import Cron
from pipeline_prefect import all_flow, code_flow, train_flow, evaluate_flow


if __name__ == "__main__":
    all_flow.serve(
        name="ml-pipeline-all",
        schedule=Cron("0 2 * * *"),   # tous les jours à 02:00
        tags=["mlops"],
    )
    train_flow.serve(name="ml-pipeline-train", tags=["mlops"])
    evaluate_flow.serve(name="ml-pipeline-evaluate", tags=["mlops"])
    code_flow.serve(name="ml-pipeline-code", tags=["mlops"])
