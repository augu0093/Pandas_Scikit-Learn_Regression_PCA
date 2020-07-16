"""
This script makes predictions for submission into the Kaggle competition.
@AugustSemrau
"""

import pandas as pd
from datetime import datetime


"""
Below function exports the predictions to .csv format for entry into the competition
"""
def csv_saver(predictions, name='Time'):

    # Get house ID's from test set
    test_path = "./data/test.csv"
    df_test = pd.read_csv(test_path)
    house_id = df_test['Id']

    # Make data frame from predictions and indexes
    df = pd.DataFrame(list(zip(house_id, predictions)), columns=['Id', 'SalePrice'], index=None)

    # Name the file the timestamp for differentiation if nothing is specified
    if name == 'Time':
        time_now = datetime.now()
        day, hour, minute = str(time_now.day), str(time_now.hour), str(time_now.minute)
        time = day + '-' + hour + '-' + minute
        name = time
    # Define output filename
    output_filename = 'predictions/predictions_{}.csv'.format(name)

    # Save to .csv
    return df.to_csv(output_filename, index=False)


