"""
Import data and save to .csv functions.
@AugustSemrau
"""


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from datetime import datetime


def dataLoader(test=False, optimize_set=False):

    # Loading both training and test set
    train_path = "./data/train.csv"
    test_path = "./data/test.csv"
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    num_train_rows = df_train.shape[0]
    num_test_rows = df_test.shape[0]

    # First separate price from training data
    X = df_train[[col for col in df_train if col not in ['SalePrice']]]
    y = df_train[['SalePrice']]

    # For pre-processing (feature engineering), both the training and test set are combined
    df_all = pd.concat([X, df_test])


    # Now all categorical variables need to be label encoded
    # Get list of categorical variables
    s = (df_all.dtypes == 'object')
    object_cols = list(s[s].index)
    # Encode all categorical variable columns
    label_df = df_all.copy()
    labelEncoder = LabelEncoder()
    for col in object_cols:
        label_df[col] = labelEncoder.fit_transform(df_all[col])
    df_all = label_df

    # Now the two data sets are separated again
    X = df_all[:num_train_rows]
    df_test = df_all[-num_test_rows:]


    # The test set is returned
    if test:
        return df_test

    # Training set is returned, either split or not
    elif not test:
        if optimize_set:
            return train_test_split(X, y, test_size=0.2, random_state=0)
        else:
            return X, y


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



