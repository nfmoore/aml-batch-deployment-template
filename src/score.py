import logging
from argparse import ArgumentParser
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from azureml.core import Run
from azureml.core.model import Model
from opencensus.ext.azure.log_exporter import AzureLogHandler

model = None


def init():
    global model

    # Parse command line arguments
    ap = ArgumentParser()
    ap.add_argument('--model_id', dest='model_id', required=True)
    args, _ = ap.parse_known_args()

    # Get model name and version
    model_name, model_version = args.model_id.split(':')

    # Retreive path to model folder
    model_path = Model.get_model_path(model_name, version=int(model_version))

    # Deserialize the model file back into a sklearn model
    model = joblib.load(model_path)


def process_data(input_df):
    # Convert strings to float
    df = input_df.astype({
        'age': np.float64, 'height': np.float64, 'weight': np.float64,
        'systolic': np.float64, 'diastolic': np.float64})

    # Define categorical / numeric features
    categorical_features = ['gender', 'cholesterol',
                            'glucose', 'smoker', 'alcoholic', 'active']
    numeric_features = ['age', 'systolic', 'diastolic', 'bmi']

    # Create feature for Body Mass Index (indicator of heart health)
    df['bmi'] = df.weight / (df.height / 100) ** 2

    # Get model features / target
    df = df.drop(labels=['height', 'weight'], axis=1)

    # Convert data types of model features
    df[categorical_features] = df[categorical_features].astype(np.object)
    df[numeric_features] = df[numeric_features].astype(np.float64)

    return df


def run(mini_batch):

    # Add the app insights logger to the python logger
    logger = logging.getLogger(__name__)
    logger.addHandler(AzureLogHandler())

    # Get run context
    run = Run.get_context()

    # Get pipeline information
    custom_dimensions = {
        'parent_run_id': run.parent.id,
        'step_id': run.id,
        'step_name': run.name,
        'experiment_name': run.experiment.name,
        'run_url': run.parent.get_portal_url(),
        'run_time': datetime.now(),
        'mini_batch': mini_batch
    }

    # Log pipeline information
    logger.info('Pipeline information', custom_dimensions)

    try:
        result_list = []

        for file_path in mini_batch:

            # Read file
            input_df = pd.read_csv(file_path)

            # Preprocess payload and get model prediction
            df = process_data(input_df)
            probability = model.predict_proba(df)

            # Add prediction and confidence level to input data as columns
            input_df['probability'] = probability[:, 1]
            input_df['score'] = (probability[:, 1] >= 0.5).astype(np.int)

            # Add scored data from file to list
            result_list.append(input_df)

        # Create a single dataframe from scored data and add datetime column
        concat_df = pd.concat(result_list)
        concat_df['score_date'] = datetime.now()

        # Log metrics to appinsights
        logger.info('Pipeline scored records', concat_df.shape[0])

        return concat_df

    except Exception as error:
        # Log exception to appinsights
        logger.error('Pipeline exception', str(error))

        # Retern exception
        return str(error)
