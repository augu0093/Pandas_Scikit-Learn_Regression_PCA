"""
This script makes predictions for submission into the Kaggle competition.
@AugustSemrau
"""

from data_loader import dataLoader
from models import Models
from data_loader import csv_saver

import pandas as pd




if __name__ == '__main__':

    # Import test data
    testData = dataLoader(test=True, optimize_set=False, return_all=False)

    # Initiate models
    Models = Models()

    # Baseline (mean) model predictions
    baseline_model = Models.build_model_baseline()
    baseline_predictions = baseline_model.predict(testData)
    csv_saver(predictions=baseline_predictions, name="baseline")

    # Linear Regression model predictions
    linear_model = Models.build_model_lr()
    linear_predictions = linear_model.predict(testData)
    # Saving predictions
    csv_saver(predictions=linear_predictions, name="linearReg")






