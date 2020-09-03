import logging

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# eh boa pratica constantes globais estarem em caps
# eh uma boa centralizar todos os seus numeros magicos
NUMERICAL_FEATURES = ["LotArea", "OverallQual", "YearBuilt"]
LABEL = "SalePrice"
DISTRIB_MEAN = 3
DISTRIB_STD = (1,)
DISTRIB_SAMPLE_SIZE = 100
N_ITER_RANDOM = 100
DATA_DIR = "../data/houses/house_dataset.csv"
PIPELINE_DIR = "artifacts/pipeline01.joblib"
SEED = 42

np.random.seed(SEED)
# https://stackoverflow.com/questions/6918493/in-python-why-use-logging-instead-of-print
logging.basicConfig(
    format="%(levelname)s-%(asctime)s:%(message)s",
    level=logging.INFO,
    datefmt="%m/%d/%Y %I:%M:%S %p",
)
logger = logging.getLogger(__name__)


def read_data(data_dir: str) -> pd.DataFrame:
    """
    É bom suas funcoes estarem com docstrings
    É bom que elas sigam um padrão de documentação
    Eu geralmente recomendo o da google
    https://google.github.io/styleguide/pyguide.html
    """
    logging.info("Reading data")
    raw_data = pd.read_csv(data_dir)
    return raw_data[NUMERICAL_FEATURES + [LABEL]]


def defines_pipeline() -> Pipeline:
    """
    This function simply defines a pipeline

    return:
        A sklearn pipeline to be fitted
    """
    imp_median = SimpleImputer(missing_values=np.nan, strategy="median")
    sc = StandardScaler()
    linear_model = Ridge()
    regr = TransformedTargetRegressor(
        regressor=linear_model, transformer=StandardScaler()
    )
    # do search
    param_dist = {
        "regressor__alpha": np.random.normal(
            loc=DISTRIB_MEAN, scale=DISTRIB_STD, size=DISTRIB_SAMPLE_SIZE
        )
    }

    search = RandomizedSearchCV(
        regr, param_distributions=param_dist, n_iter=N_ITER_RANDOM
    )
    steps = [
        ("input_median", imp_median),
        ("standard_scaler", sc),
        ("linear_regression", search),
    ]
    pipeline = Pipeline(steps)
    return pipeline


def splits_train_saves(df: pd.DataFrame, pipeline: Pipeline, save_dir: str) -> None:
    X_train, X_test, y_train, y_test = train_test_split(
        df[NUMERICAL_FEATURES], df[[LABEL]], random_state=SEED
    )
    logging.info("Performing training")
    pipeline.fit(X_train, y_train)
    logging.info("Training completeded")
    result_on_test = r2_score(y_test, pipeline.predict(X_test))
    logging.info(f"Model obtained a score of {result_on_test} R2 on test set")
    pipeline_saved = dump(pipeline, save_dir)
    logging.info(f"Pipeline saved in {pipeline_saved[0]}")


if __name__ == "__main__":
    current_pipeline = defines_pipeline()
    df = read_data(DATA_DIR)
    splits_train_saves(df, current_pipeline, PIPELINE_DIR)
