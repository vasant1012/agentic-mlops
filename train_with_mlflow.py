import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from utils.logger import logger
import random


def run_experiment(C: float, max_iter: int):
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.3, random_state=42
    )
    with mlflow.start_run() as run:
        # log params
        mlflow.log_param("C", C)
        mlflow.log_param("max_iter", max_iter)

        model = LogisticRegression(C=C, max_iter=max_iter)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        loss = log_loss(y_test, y_proba)

        # log metrics
        logger.info(f'Accuracy for current experiment is{acc}')
        mlflow.log_metric("accuracy", acc)
        logger.info(f'Log_loss for current experiment is{loss}')
        mlflow.log_metric("log_loss", loss)

        # log model
        mlflow.sklearn.log_model(model, "model")

        logger.info(f"Run {run.info.run_id} | acc={acc:.3f} | loss={loss:.3f}")


if __name__ == "__main__":
    experiment_name = "iris_demo"
    mlflow.set_experiment(experiment_name)
    logger.info(f'The name of experiment is {experiment_name}')

    # Generate multiple runs with random hyperparams
    for _ in range(5):
        C = random.choice([0.01, 0.1, 1.0, 10.0])
        max_iter = random.choice([100, 200, 300])
        run_experiment(C, max_iter)
