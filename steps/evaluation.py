import logging

import pandas as pd
from sklearn.base import RegressorMixin
from typing import Tuple
from typing_extensions import Annotated
from zenml import step

from src.evaluation import MSE, R2, RMSE

@step
def evaluate_model(
        model: RegressorMixin,
        X_test: pd.DataFrame,
        y_test: pd.DataFrame,
) -> Tuple[
    Annotated[float, "R2 score"],
    Annotated[float, "RMSE score"]
]:
    """
    Evaluates the model on the ingested data

    Args:
        model: Trained model
        X_test: Test data
        y_test: Test labels
    Returns:
        r2_score: R2 score
        rmse_score: RMSE score
    """
    try:
        prediction = model.predict(X_test)
        mse_class = MSE()
        mse_score = mse_class.calculate_score(y_test, prediction)

        r2_class = R2()
        r2_score = r2_class.calculate_score(y_test, prediction)

        rmse_class = RMSE()
        rmse_score = rmse_class.calculate_score(y_test, prediction)

        return r2_score, rmse_score
    except Exception as e:
        logging.error("Error in evaluating model: {}".format(e))
        raise e