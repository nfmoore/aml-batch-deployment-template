from argparse import ArgumentParser

import joblib
import numpy as np
from azureml.core.model import Model

model = None
# inputs_dc = None
# prediction_dc = None


def init():
    # from azureml.monitoring import ModelDataCollector

    global model
    global inputs_dc, prediction_dc

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

    # Initialize data collectors
    # inputs_dc = ModelDataCollector(
    #     model_name='cardiovascular_disease_model',
    #     designation='inputs',
    #     feature_names=['age', 'gender', 'systolic', 'diastolic', 'height',
    #                    'weight', 'cholesterol', 'glucose', 'smoker',
    #                    'alcoholic', 'active'])
    # prediction_dc = ModelDataCollector(
    #     model_name="cardiovascular_disease_model",
    #     designation='predictions',
    #     feature_names=['cardiovascular_disease'])


def process_data(input_df):
    # Convert strings to float
    df = input_df.astype({
        'age': np.float64, 'height': np.float64, 'weight': np.float64,
        'systolic': np.float64, 'diastolic': np.float64,
        'cardiovascular_disease': np.float64})

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


def run(input_df):
    try:
        print('input_df', type(input_df), input_df.shape)
        # Preprocess payload and get model prediction
        df = process_data(input_df)
        print('X', df)
        probability = model.predict_proba(df)
        print('probability', probability)
        input_df['probability'] = probability[:, 1]
        input_df['score'] = (probability[:, 1] >= 0.5).astype(np.int)

        # Log input and prediction to appinsights
        # print('Request Payload', data)
        # print('Response Payload', result)

        # Collect features and prediction data
        # inputs_dc.collect(input_df)
        # prediction_dc.collect(pd.DataFrame((proba[:, 1] >= 0.5).astype(int),
        #                                    columns=['cardiovascular_disease']))

        return input_df

    except Exception as error:
        # Log exception to appinsights
        # print('Error', str(e))

        # Retern exception
        return str(error)
