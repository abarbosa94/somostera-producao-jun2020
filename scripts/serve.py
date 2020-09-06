import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
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
app = Flask("producao-modelo")


def load_model(pipeline_dir: str) -> Pipeline:
    pipeline = load(pipeline_dir)
    return pipeline


def run_pipeline(pipeline: Pipeline, **features: dict) -> float:
    diff = set(NUMERICAL_FEATURES) - features.keys()
    if len(diff) != 0:
        return -1
    datapoint = pd.DataFrame.from_dict(features, orient="index").T
    datapoint = datapoint[NUMERICAL_FEATURES]  # garantir ordem!
    result = pipeline.predict(datapoint)[0][0]
    logging.info(f"This house should have {result:.2f} dollars as price")
    return result

@app.route('/predict', methods=['POST'])
def predict():
    full_pipe = load_model(PIPELINE_DIR)
    json_ = request.json
    result = run_pipeline(full_pipe, **json_)
    if result == -1:
        logging.error("Please review your passed features")
        result = None
    return jsonify(result)

@app.route('/', methods=['GET'])
def hello_world():
    return "Hello world"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    from waitress import serve
    serve(app, host="0.0.0.0", port=port)
