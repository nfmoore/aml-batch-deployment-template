from unittest.mock import MagicMock, patch

import pandas as pd
from tests.fixtures import data
from tests.mocks import MockModel

from src.score import process_data, run


def test_process_data():
    # Generate payload
    payload = data.copy()

    # Apply preprocessing
    input_df = pd.DataFrame(payload)
    input_df.drop('cardiovascular_disease', axis=1, inplace=True)
    X = process_data(input_df)

    # Calculate BMI value
    payload_bmi = input_df.iloc[0]['weight'] / \
        (input_df.iloc[0]['height'] / 100) ** 2

    # Should return a dataframe with 1 row and 10 columns
    print('X.columns', X.columns)
    assert X.shape == (input_df.shape[0], 10)

    # Should include column for BMI
    assert 'bmi' in X.columns.tolist()

    # Should remove height and weight columns
    assert 'height' not in X.columns.tolist()
    assert 'weight' not in X.columns.tolist()

    # Should contain correct BMI value
    assert X.iloc[0].bmi == payload_bmi


@patch('src.score.model', MockModel())
@patch('src.score.logging', MagicMock())
@patch('src.score.AzureLogHandler', MagicMock())
@patch('src.score.Run.get_context', MagicMock())
@patch('src.score.pd.read_csv')
def test_run(mock_pd_read_csv):
    # Generate mini_batch
    mini_batch = pd.DataFrame(data.copy())
    mini_batch.drop('cardiovascular_disease', axis=1, inplace=True)
    mock_pd_read_csv.return_value = mini_batch

    # Return prediction
    result = run([mini_batch])
    print('result', result)
    # Should return valid response payload
    assert 'probability' in result.columns
    assert 'score' in result.columns
    assert 'score_date' in result.columns
    assert type(result) == type(mini_batch)
