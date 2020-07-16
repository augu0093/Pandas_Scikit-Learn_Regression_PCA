"""
Import data and save to .csv functions.
@AugustSemrau
"""


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split



"""
Feature engineering of the 79 explanatory variables of the house pricing data set
"""
def feat_eng(data, drop=False):

    # 1.Imputation



    # 2. Handling Outliers


    # 3.Binning


    # 4.Log Transform


    # 5.One-Hot Encoding

    # # Now all categorical variables need to be label encoded
    # # Get list of categorical variables
    # s = (data.dtypes == 'object')
    # object_cols = list(s[s].index)
    # # Encode all categorical variable columns
    # label_df = data.copy()
    # label_encoder = LabelEncoder()
    # for col in object_cols:
    #     label_df[col] = label_encoder.fit_transform(data[col])
    # df_all = label_df

    # 6.Grouping Operations


    # 7.Feature Split


    # 8.Scaling


    # 9.Extracting Date


    return data


"""
Data loading is performed in below function
"""
def dataLoader(test=False, optimize_set=False, return_all=False):

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

    # Feature engineering is performed in separate function
    df_all = feat_eng(df_all)
    if return_all:
        return df_all

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




"""
Checking the dimensions of the loaded data
"""
if __name__ == '__main__':

    all_data = dataLoader(return_all=True)
    print('All feature data')
    print(all_data.dtypes)
    print(all_data.isna().sum())
    print(all_data.shape)
    print('')

    X, y = dataLoader(test=False, optimize_set=False)
    print('Training set')
    print(X.shape)
    print(y.shape)
    print('')

    X_train, X_val, y_train, y_val = dataLoader(test=False, optimize_set=True)
    print('Split training set')
    print(X_train.shape)
    print(y_train.shape)
    print(X_val.shape)
    print(y_val.shape)
    print('')

    test_data = dataLoader(test=True, optimize_set=False)
    print('Test data for predictions')
    print(test_data.shape)

