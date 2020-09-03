import argparse
import logging
import sys

import numpy as np
import pandas as pd
from joblib import load
from sklearn.pipeline import Pipeline

NUMERICAL_FEATURES = ["LotArea", "OverallQual", "YearBuilt"]
PIPELINE_DIR = "artifacts/pipeline01.joblib"
SEED = 42

np.random.seed(SEED)
logging.basicConfig(
    format="%(levelname)s-%(asctime)s:%(message)s",
    level=logging.INFO,
    datefmt="%m/%d/%Y %I:%M:%S %p",
)
logger = logging.getLogger(__name__)


def load_model(pipeline_dir: str) -> Pipeline:
    pipeline = load(pipeline_dir)
    return pipeline


def run_pipeline(pipeline: Pipeline, **features: dict) -> float:
    diff = set(NUMERICAL_FEATURES) - features.keys()
    if len(diff) != 0:
        logging.error(
            "Please review your passed features, the different set" f"was set by {diff}"
        )
        sys.exit("Check the feature names that you have passed!")
        return -1
    datapoint = pd.DataFrame.from_dict(features, orient="index").T
    datapoint = datapoint[NUMERICAL_FEATURES]  # garantir ordem!
    result = pipeline.predict(datapoint)[0][0]
    logging.info(f"This house should have {result:.2f} dollars as price")
    return 1


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Define features of a datapoint")
    parser.add_argument("--lot_area", type=float, help="House Area", required=True)
    parser.add_argument("--year", type=int, help="House Built Year", required=True)
    parser.add_argument(
        "--quality", type=int, help="Overall Quality House", required=True
    )

    args = parser.parse_args()
    features = {
        "LotArea": args.lot_area,
        "YearBuilt": args.year,
        "OverallQual": args.quality,
    }
    run_pipeline(load_model(PIPELINE_DIR), **features)
