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
    # Getting features that need imputation
    impute_feature_keys = list(data.columns)
    impute_feature_vals = data.isna().sum().tolist()
    impute_dict = dict(zip(impute_feature_keys, impute_feature_vals))
    impute_dict = {key: val for key, val in impute_dict.items() if val != 0}

    # Some features have 1-2 missing values and are removed
    remove_feats = {key: val for key, val in impute_dict.items() if val <= 4}
    data.dropna(subset=remove_feats.keys(), inplace=True)

    # These features were:
    # {'MSZoning': 4,       # Identifies the general zoning classification of the sale.
    # 'Utilities': 2,       # Type of utilities available
    # 'Exterior1st': 1,     # Exterior covering on house
    # 'Exterior2nd': 1,     # Exterior covering on house (if more than one material)
    # 'BsmtFinSF1': 1,      # Type 1 finished square feet ((Basement))
    # 'BsmtFinSF2': 1,      # Type 2 finished square feet ((Basement))
    # 'BsmtUnfSF': 1,       # Unfinished square feet of basement area
    # 'TotalBsmtSF': 1,     # Total square feet of basement area
    # 'Electrical': 1,      # Electrical system
    # 'BsmtFullBath': 2,    # Basement full bathrooms
    # 'BsmtHalfBath': 2,    # Basement half bathrooms
    # 'KitchenQual': 1,     # Kitchen quality
    # 'Functional': 2,      # Home functionality (Assume typical unless deductions are warranted)
    # 'GarageCars': 1,      # Size of garage in car capacity
    # 'GarageArea': 1,      # Size of garage in square feet
    # 'SaleType': 1}        # Type of sale

    # Some features are NA because they don't exist, e.g. no fireplace gives no fireplace quality, imputed to 0
    # Redefining impute dict for non-removed features
    impute_dict = {key: val for key, val in impute_dict.items() if val > 4}

    # These features are following:
    # {'LotFrontage': 484,      # Linear feet of street connected to property
    #  'Alley': 2709,           # Type of alley access to property: NA 	No alley access
    #  'MasVnrType': 24,        # Masonry veneer type
    #  'MasVnrArea': 23,        # Masonry veneer area in square feet
    #  'BsmtQual': 76,          # Evaluates the height of the basement: NA	No Basement
    #  'BsmtCond': 77,          # Evaluates the general condition of the basement: NA	No Basement
    #  'BsmtExposure': 77,      # Refers to walkout or garden level walls: NA	No Basement
    #  'BsmtFinType1': 74,      # Rating of basement finished area: NA	No Basement
    #  'BsmtFinType2': 75,      # Rating of basement finished area (if multiple types): NA	No Basement
    #  'FireplaceQu': 1411,     # Fireplace quality: NA	No Fireplace
    #  'GarageType': 156,       # Garage location: NA	No Garage
    #  'GarageYrBlt': 157,      # Year garage was built
    #  'GarageFinish': 157,     # Interior finish of the garage: NA	No Garage
    #  'GarageQual': 157,       # Garage quality: NA	No Garage
    #  'GarageCond': 157,       # Garage condition: NA	No Garage
    #  'PoolQC': 2896,          # Pool quality: NA	No Pool
    #  'Fence': 2337,           # Fence quality: NA	No Fence
    #  'MiscFeature': 2802}     # Miscellaneous feature not covered in other categories: NA	None

    # For the above features where NA means non-existing, these are changed to 'NON'
    other_keys = {'LotFrontage', 'MasVnrType', 'MasVnrArea', 'GarageYrBlt'}
    none_dict = {x: impute_dict[x] for x in impute_dict if x not in other_keys}
    none_list = none_dict.keys()
    for feat in none_list:
        data[feat].fillna('NOTHING', inplace=True)

    # The last four features are imputed as follows:
    # LotFrontage is imputed to the mean
    data['LotFrontage'].fillna(data['LotFrontage'].mean(), inplace=True)
    # MasVnrType is imputed to 'None' and MasVnrArea is imputed to 0
    data['MasVnrType'].fillna('None', inplace=True)
    data['MasVnrArea'].fillna(0, inplace=True)
    # GarageYrBlt is imputed to mean year
    data['GarageYrBlt'].fillna(data['GarageYrBlt'].mean().round(), inplace=True)


    # 2. Handling Outliers


    # 3. Binning


    # 4. Log Transform


    # 5. One-Hot Encoding

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

    # 6. Grouping Operations


    # 7. Feature Split


    # 8. Scaling


    # 9. Extracting Date

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
    feature_names = list(all_data.columns)
    number_na = all_data.isna().sum().tolist()
    nas = dict(zip(feature_names, number_na))
    nas = {key:val for key, val in nas.items() if val != 0}
    print(nas)
    print(len(nas))
    print(all_data.shape)
    print('')

    # X, y = dataLoader(test=False, optimize_set=False)
    # print('Training set')
    # print(X.shape)
    # print(y.shape)
    # print('')
    #
    # X_train, X_val, y_train, y_val = dataLoader(test=False, optimize_set=True)
    # print('Split training set')
    # print(X_train.shape)
    # print(y_train.shape)
    # print(X_val.shape)
    # print(y_val.shape)
    # print('')
    #
    # test_data = dataLoader(test=True, optimize_set=False)
    # print('Test data for predictions')
    # print(test_data.shape)

