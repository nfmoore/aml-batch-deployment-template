import numpy as np
import pandas as pd
from pytest import fixture

records = [
    {
        "age": 50.391780821917806,
        "gender": "female",
        "height": 168,
        "weight": 62.0,
        "systolic": 110,
        "diastolic": 80,
        "cholesterol": "normal",
        "glucose": "normal",
        "smoker": "not-smoker",
        "alcoholic": "not-alcoholic",
        "active": "active",
        "cardiovascular_disease": 0,
    },
    {
        "age": 55.41917808219178,
        "gender": "male",
        "height": 156,
        "weight": 85.0,
        "systolic": 140,
        "diastolic": 90,
        "cholesterol": "well-above-normal",
        "glucose": "normal",
        "smoker": "not-smoker",
        "alcoholic": "not-alcoholic",
        "active": "active",
        "cardiovascular_disease": 1,
    },
    {
        "age": 51.66301369863014,
        "gender": "male",
        "height": 165,
        "weight": 64.0,
        "systolic": 130,
        "diastolic": 70,
        "cholesterol": "well-above-normal",
        "glucose": "normal",
        "smoker": "not-smoker",
        "alcoholic": "not-alcoholic",
        "active": "not-active",
        "cardiovascular_disease": 1,
    },
    {
        "age": 48.28219178082192,
        "gender": "female",
        "height": 169,
        "weight": 82.0,
        "systolic": 150,
        "diastolic": 100,
        "cholesterol": "normal",
        "glucose": "normal",
        "smoker": "not-smoker",
        "alcoholic": "not-alcoholic",
        "active": "active",
        "cardiovascular_disease": 1,
    },
    {
        "age": 47.87397260273973,
        "gender": "male",
        "height": 156,
        "weight": 56.0,
        "systolic": 100,
        "diastolic": 60,
        "cholesterol": "normal",
        "glucose": "normal",
        "smoker": "not-smoker",
        "alcoholic": "not-alcoholic",
        "active": "not-active",
        "cardiovascular_disease": 0,
    },
    {
        "age": 60.038356164383565,
        "gender": "male",
        "height": 151,
        "weight": 67.0,
        "systolic": 120,
        "diastolic": 80,
        "cholesterol": "above-normal",
        "glucose": "above-normal",
        "smoker": "not-smoker",
        "alcoholic": "not-alcoholic",
        "active": "not-active",
        "cardiovascular_disease": 0,
    },
    {
        "age": 60.583561643835615,
        "gender": "male",
        "height": 157,
        "weight": 93.0,
        "systolic": 130,
        "diastolic": 80,
        "cholesterol": "well-above-normal",
        "glucose": "normal",
        "smoker": "not-smoker",
        "alcoholic": "not-alcoholic",
        "active": "active",
        "cardiovascular_disease": 0,
    },
    {
        "age": 61.87397260273973,
        "gender": "female",
        "height": 178,
        "weight": 95.0,
        "systolic": 130,
        "diastolic": 90,
        "cholesterol": "well-above-normal",
        "glucose": "well-above-normal",
        "smoker": "not-smoker",
        "alcoholic": "not-alcoholic",
        "active": "active",
        "cardiovascular_disease": 1,
    },
]


@fixture
def data():
    return records


@fixture
def input_df():
    return pd.DataFrame(records)


@fixture
def cv_results():
    return {
        "train_score": np.array([0.74, 0.70, 0.72, 0.71]),
        "test_score": np.array([0.73, 0.71, 0.73, 0.72]),
    }
